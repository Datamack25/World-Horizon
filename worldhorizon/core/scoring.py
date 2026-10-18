"""Agent Synthèse — normalisation des indicateurs, scores par pilier, score actuel et historique.

Méthode (transparente et reproductible) :
 1. Pour chaque indicateur, on prend la dernière valeur disponible (au plus 8 ans d'ancienneté
    pour le score actuel, 5 ans pour l'historique annuel).
 2. Transformation éventuelle (log, distance à une cible comme 2 % d'inflation, rapport à la population).
 3. Normalisation 0-100 sur des bornes fixes = 5e et 95e centiles de la distribution mondiale actuelle
    (les mêmes bornes servent pour toutes les années : les évolutions historiques sont comparables).
 4. Score de pilier = moyenne pondérée des indicateurs disponibles (si couverture ≥ 35 %).
 5. Score actuel = moyenne pondérée des piliers disponibles (au moins 6 piliers sur 14 ;
    l'indice de fiabilité = part des piliers et indicateurs effectivement renseignés).
    Les scores PISA ne comptent dans la couverture que pour les pays qui participent à PISA.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .catalog import AGENTS, COLOR_THRESHOLDS, pc_scale
from .paths import SCORES

YEAR0 = 1970
CURRENT_YEAR = dt.date.today().year
YEARS = list(range(YEAR0, CURRENT_YEAR + 1))
MIN_PILLAR_COVERAGE = 0.35
MIN_PILLARS = 6  # en dessous : « données insuffisantes » ; la fiabilité est affichée


def _wide(long: pd.DataFrame, code: str) -> pd.DataFrame:
    d = long[long["indicator"] == code]
    if d.empty:
        return pd.DataFrame(columns=YEARS, dtype=float)
    w = d.pivot_table(index="iso3", columns="year", values="value", aggfunc="last")
    return w.reindex(columns=YEARS)


def population_wide(long: pd.DataFrame) -> pd.DataFrame:
    p = _wide(long, "sp_pop_totl")
    gm = _wide(long, "pop")
    return p.combine_first(gm).ffill(axis=1, limit=3)


def transform(values: pd.DataFrame | pd.Series, ind: dict):
    v = values.astype(float)
    t = ind.get("transform")
    if t == "log":
        v = np.log10(v.clip(lower=1e-3))
    elif t and t.startswith("target:"):
        v = (v - float(t.split(":")[1])).abs()
    return v


class Scorer:
    """Construit, pour chaque indicateur noté, la matrice pays × année des valeurs
    (avec report des dernières valeurs connues) et les scores 0-100."""

    def __init__(self, long: pd.DataFrame, countries: pd.DataFrame):
        self.long = long
        self.countries = countries.set_index("iso3")
        self.iso = list(countries["iso3"])
        self.pop = population_wide(long).reindex(self.iso)
        self.raw: dict[str, pd.DataFrame] = {}
        self.raw_hist: dict[str, pd.DataFrame] = {}
        self.score: dict[str, pd.DataFrame] = {}
        self.score_now: dict[str, pd.Series] = {}
        self.value_now: dict[str, pd.Series] = {}
        self.year_now: dict[str, pd.Series] = {}
        self.goalposts = []
        self._build()

    def _build(self):
        seen = set()
        for agent in AGENTS:
            for ind in agent["indicateurs"]:
                key = ind["code"]
                if key in seen or ind["weight"] == 0:
                    continue
                seen.add(key)
                w = _wide(self.long, key).reindex(self.iso)
                if pc_scale(ind):
                    w = w / self.pop * pc_scale(ind)
                self.raw[key] = w
                # dernière valeur connue (≤ 8 ans) pour le score actuel
                filled = w.ffill(axis=1, limit=8)
                last_year = w.apply(lambda r: r.last_valid_index(), axis=1)
                now = filled[CURRENT_YEAR]
                self.value_now[key] = now
                self.year_now[key] = last_year.where(now.notna())
                t_now = transform(now, ind)
                lo, hi = np.nanpercentile(t_now.dropna(), [5, 95]) if t_now.notna().sum() > 10 else (np.nan, np.nan)
                if not np.isfinite(lo) or hi == lo:
                    continue
                self.goalposts.append(dict(indicator=key, label=ind["label"], lo=lo, hi=hi,
                                           direction=ind["direction"], transform=ind.get("transform") or ""))
                hist = transform(w.ffill(axis=1, limit=5), ind)
                s = ((hist - lo) / (hi - lo)).clip(0, 1) * 100
                if ind["direction"] < 0:
                    s = 100 - s
                self.score[key] = s
                sn = ((t_now - lo) / (hi - lo)).clip(0, 1) * 100
                self.score_now[key] = sn if ind["direction"] > 0 else 100 - sn

    # ---------------------------------------------------------------- piliers
    def pillar_now(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        scores, cover = {}, {}
        for a in AGENTS:
            num = pd.Series(0.0, index=self.iso); den = pd.Series(0.0, index=self.iso)
            tot = pd.Series(0.0, index=self.iso)
            for ind in a["indicateurs"]:
                if ind["weight"] == 0 or ind["code"] not in self.score_now:
                    continue
                s = self.score_now[ind["code"]]
                if ind["src"] == "pisa":
                    tot += s.notna() * ind["weight"]
                else:
                    tot += ind["weight"]
                num += s.fillna(0) * ind["weight"]
                den += s.notna() * ind["weight"]
            cov = den / tot.replace(0, np.nan)
            sc = (num / den.replace(0, np.nan)).where(cov >= MIN_PILLAR_COVERAGE)
            scores[a["id"]] = sc; cover[a["id"]] = cov
        return pd.DataFrame(scores), pd.DataFrame(cover)

    def pillar_history(self) -> pd.DataFrame:
        """Score de chaque pilier pour chaque année (format long)."""
        out = []
        for a in AGENTS:
            num = None; den = None; tot = 0.0
            for ind in a["indicateurs"]:
                if ind["weight"] == 0 or ind["code"] not in self.score or ind["src"] == "pisa":
                    continue
                s = self.score[ind["code"]]
                tot += ind["weight"]
                num = s.fillna(0) * ind["weight"] if num is None else num + s.fillna(0) * ind["weight"]
                den = s.notna() * ind["weight"] if den is None else den + s.notna() * ind["weight"]
            if num is None:
                continue
            sc = (num / den.replace(0, np.nan)).where(den / tot >= MIN_PILLAR_COVERAGE)
            m = sc.stack().rename("score").reset_index()
            m.columns = ["iso3", "year", "score"]
            m["pillar"] = a["id"]
            out.append(m)
        return pd.concat(out, ignore_index=True)


def aggregate(pillars: pd.DataFrame) -> pd.Series:
    w = pd.Series({a["id"]: a["poids"] for a in AGENTS})
    p = pillars[w.index]
    num = (p * w).sum(axis=1, min_count=1)
    den = p.notna().mul(w).sum(axis=1)
    sc = num / den.replace(0, np.nan)
    return sc.where(p.notna().sum(axis=1) >= MIN_PILLARS)


def color_of(score: float) -> str:
    if score is None or not np.isfinite(score):
        return "gris"
    if score >= COLOR_THRESHOLDS["vert"]:
        return "vert"
    if score >= COLOR_THRESHOLDS["orange"]:
        return "orange"
    return "rouge"


LEVELS = [(75, "très favorable"), (60, "favorable"), (45, "intermédiaire"), (30, "fragile"), (-1, "critique")]


def level_of(score: float) -> str:
    if score is None or not np.isfinite(score):
        return "non évalué"
    return next(l for t, l in LEVELS if score >= t)


def compute_scores(long: pd.DataFrame, countries: pd.DataFrame) -> dict:
    sc = Scorer(long, countries)
    pillars, cover = sc.pillar_now()
    current = aggregate(pillars)
    reliability = (pillars.notna().sum(axis=1) / len(AGENTS) * 0.5 + cover.clip(upper=1).mean(axis=1) * 0.5) * 100
    hist = sc.pillar_history()
    hw = hist.pivot_table(index=["iso3", "year"], columns="pillar", values="score")
    for a in AGENTS:
        if a["id"] not in hw:
            hw[a["id"]] = np.nan
    hist_global = aggregate(hw).rename("score_actuel").reset_index()

    # indicateurs (valeur, année, score) pour l'état actuel
    rows = []
    for a in AGENTS:
        for ind in a["indicateurs"]:
            k = ind["code"]
            if k not in sc.value_now:
                continue
            df = pd.DataFrame({"iso3": sc.iso, "pillar": a["id"], "indicator": k,
                               "value": sc.value_now[k].values, "year": sc.year_now[k].values,
                               "score": sc.score_now.get(k, pd.Series(np.nan, index=sc.iso)).values})
            rows.append(df)
    ind_now = pd.concat(rows, ignore_index=True).dropna(subset=["value"])

    pillars.index.name = "iso3"; cover.index.name = "iso3"
    pillars.round(2).to_csv(SCORES / "pillars_current.csv")
    cover.round(3).to_csv(SCORES / "coverage_current.csv")
    hist.round(2).to_csv(SCORES / "pillars_history.csv", index=False)
    hist_global.dropna().round(2).to_csv(SCORES / "global_history.csv", index=False)
    ind_now.round(4).to_csv(SCORES / "indicators_current.csv", index=False)
    pd.DataFrame(sc.goalposts).to_csv(SCORES / "goalposts.csv", index=False)
    return dict(scorer=sc, pillars=pillars, coverage=cover, current=current, reliability=reliability, history=hist,
                global_history=hist_global, indicators=ind_now)
