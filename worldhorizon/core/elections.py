"""Agent Pronostics électoraux & risque politique.

- Indice de risque politique (0-100) = 45 % probabilité ML de recul démocratique à 3 ans
  (en rang centile mondial) + 55 % fragilité structurelle (sécurité, impunité, corruption,
  économie, bien-être social, polarisation).
- Agrégateur de sondages : moyenne pondérée par récence (demi-vie 14 jours) et taille
  d'échantillon, puis simulation Monte-Carlo (erreur historique moyenne des sondages ≈ 3,5 pts,
  plus une erreur corrélée commune) → probabilité d'arriver en tête.
- Calendrier : data/elections/calendar.csv (modifiable dans l'application), avec tentative
  de mise à jour depuis Wikipédia (calendrier électoral national).
- Simulateur « what-if » : recalcul instantané du score si un ou plusieurs piliers évoluent.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .catalog import AGENTS
from .paths import ELECTIONS, ML_DIR

CAL = ELECTIONS / "calendar.csv"
POLLS = ELECTIONS / "polls.csv"
POLL_ERROR_SD = 3.5
COMMON_ERROR_SD = 1.5


def political_risk_index(pillars: pd.DataFrame, risk: pd.DataFrame, long: pd.DataFrame | None = None) -> pd.DataFrame:
    ml = risk["proba_choc_3ans"].reindex(pillars.index)
    ml_pct = ml.rank(pct=True) * 100
    frag_cols = ["securite", "impunite", "corruption", "economie", "social"]
    fragility = (100 - pillars[frag_cols].mean(axis=1)).where(pillars[frag_cols].notna().sum(axis=1) >= 3)
    polar = pd.Series(np.nan, index=pillars.index)
    if long is not None:
        p = long[long.indicator == "v2cacamps"].sort_values("year").groupby("iso3")["value"].last()
        polar = (p.rank(pct=True) * 100).reindex(pillars.index)
    struct = (0.85 * fragility + 0.15 * polar.fillna(50))
    idx = (0.45 * ml_pct.fillna(ml_pct.median()) + 0.55 * struct).where(fragility.notna())
    out = pd.DataFrame({"indice_risque_politique": idx, "proba_recul_democratique_3ans": ml,
                        "fragilite_structurelle": fragility, "polarisation_centile": polar})
    out["niveau"] = pd.cut(out["indice_risque_politique"], [-1, 40, 60, 101], labels=["Faible", "Modéré", "Élevé"])
    out.index.name = "iso3"
    out.round(3).to_csv(ML_DIR / "political_risk_index.csv")
    return out


def load_calendar() -> pd.DataFrame:
    if not CAL.exists():
        return pd.DataFrame(columns=["iso3", "date", "election", "type", "statut", "source"])
    c = pd.read_csv(CAL)
    c["date"] = pd.to_datetime(c["date"], errors="coerce")
    return c.sort_values("date")


def upcoming(cal: pd.DataFrame, today: dt.date | None = None) -> pd.DataFrame:
    today = pd.Timestamp(today or dt.date.today())
    return cal[cal["date"] >= today - pd.Timedelta(days=1)]


def load_polls() -> pd.DataFrame:
    if not POLLS.exists():
        return pd.DataFrame(columns=["iso3", "election", "date_sondage", "institut", "echantillon",
                                     "candidat_ou_parti", "intention_pct", "source"])
    p = pd.read_csv(POLLS)
    p["date_sondage"] = pd.to_datetime(p["date_sondage"], errors="coerce")
    return p


def aggregate_polls(polls: pd.DataFrame, iso3: str, election: str | None = None,
                    window_days: int = 90, n_sims: int = 20000, seed: int = 7) -> dict | None:
    p = polls[polls["iso3"] == iso3].dropna(subset=["intention_pct", "date_sondage"])
    if election:
        p = p[p["election"] == election]
    if p.empty:
        return None
    last = p["date_sondage"].max()
    p = p[p["date_sondage"] >= last - pd.Timedelta(days=window_days)].copy()
    age = (last - p["date_sondage"]).dt.days
    n = pd.to_numeric(p["echantillon"], errors="coerce").fillna(1000).clip(200, 10000)
    p["w"] = np.exp(-np.log(2) * age / 14) * np.sqrt(n / 1000)
    agg = p.groupby("candidat_ou_parti").apply(lambda g: np.average(g["intention_pct"], weights=g["w"]), include_groups=False)
    agg = agg.sort_values(ascending=False)
    rng = np.random.default_rng(seed)
    k = len(agg)
    sims = agg.values[None, :] + rng.normal(0, POLL_ERROR_SD, (n_sims, k)) + rng.normal(0, COMMON_ERROR_SD, (n_sims, 1)) * rng.choice([-1, 1], k)[None, :]
    winners = sims.argmax(1)
    prob = pd.Series(np.bincount(winners, minlength=k) / n_sims, index=agg.index)
    lo = pd.Series(np.percentile(sims, 10, axis=0), index=agg.index)
    hi = pd.Series(np.percentile(sims, 90, axis=0), index=agg.index)
    res = pd.DataFrame({"moyenne_ponderee": agg.round(1), "intervalle_80_bas": lo.round(1),
                        "intervalle_80_haut": hi.round(1), "proba_en_tete": prob.round(3)})
    return dict(table=res, n_polls=int(p[["institut", "date_sondage"]].drop_duplicates().shape[0]),
                last_poll=last.date(), window_days=window_days)


def what_if(pillars_row: pd.Series, changes: dict[str, float]) -> float:
    """Score actuel recalculé après modification de certains piliers (en points)."""
    w = pd.Series({a["id"]: a["poids"] for a in AGENTS})
    p = pillars_row.reindex(w.index).astype(float).copy()
    for k, v in changes.items():
        if k in p and pd.notna(p[k]):
            p[k] = float(np.clip(p[k] + v, 0, 100))
    m = p.notna()
    return float((p[m] * w[m]).sum() / w[m].sum()) if m.any() else float("nan")


def refresh_calendar_from_wikipedia(countries: pd.DataFrame, years=None, log=print) -> int:
    """Best effort : lit les pages « <année> national electoral calendar » de Wikipédia
    et ajoute les élections nationales manquantes. Retourne le nombre de lignes ajoutées."""
    years = years or [dt.date.today().year, dt.date.today().year + 1]
    try:
        import requests
        from io import StringIO
    except ImportError:
        return 0
    name_to_iso = {str(n).lower(): i for n, i in zip(countries["name_en"], countries["iso3"])}
    cal = pd.read_csv(CAL) if CAL.exists() else pd.DataFrame(columns=["iso3", "date", "election", "type", "statut", "source"])
    added = 0
    for y in years:
        url = f"https://en.wikipedia.org/wiki/{y}_national_electoral_calendar"
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "WorldHorizon/1.0"})
            if r.status_code != 200:
                continue
            tables = pd.read_html(StringIO(r.text))
        except Exception as e:  # lxml absent, réseau, structure de page…
            log(f"Calendrier Wikipédia {y} non lu : {e}")
            continue
        import re
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if not any("country" in c for c in cols):
                continue
            t.columns = cols
            ccol = next(c for c in cols if "country" in c)
            dcol = next((c for c in cols if "date" in c), None)
            ecol = next((c for c in cols if "election" in c or "type" in c), None)
            for _, row in t.iterrows():
                iso = name_to_iso.get(str(row[ccol]).split("[")[0].strip().lower())
                if not iso:
                    continue
                dtxt = str(row[dcol]) if dcol else ""
                d = pd.to_datetime(re.sub(r"\[.*?\]", "", f"{dtxt} {y}" if str(y) not in dtxt else dtxt), errors="coerce")
                if pd.isna(d):
                    continue
                el = str(row[ecol]) if ecol else "Élection nationale"
                dup = (cal["iso3"] == iso) & (pd.to_datetime(cal["date"], errors="coerce") == d)
                if dup.any():
                    continue
                cal.loc[len(cal)] = [iso, d.date().isoformat(), el, "nationale", "Wikipédia — à vérifier", url]
                added += 1
    if added:
        cal.to_csv(CAL, index=False)
    return added
