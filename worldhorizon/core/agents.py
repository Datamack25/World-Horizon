"""Agents de collecte : chaque agent thématique récupère ses séries de façon indépendante
et alimente son propre CSV historique dans data/agents/<agent>.csv (format long).

Colonnes : iso3, year, indicator, value, src  (origine et date de collecte dans _runs_log.csv)
Les nouvelles valeurs remplacent les anciennes pour la même (iso3, année, indicateur) ;
les séries qui ne sont plus publiées restent conservées.
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .catalog import AGENTS, AUX_INDICATORS
from .paths import AGENTS_DIR, REF
from .sources import fetch_series

START_YEAR = 1970
RUN_LOG = AGENTS_DIR / "_runs_log.csv"


def _fetch_indicator(ind, valid_iso3: set[str]):
    if ind["src"] == "pisa":
        p = pd.read_csv(REF / "pisa.csv")
        df = p[["iso3", "year", ind["code"]]].rename(columns={ind["code"]: "value"})
        origin = "data/ref/pisa.csv (OCDE PISA 2022)"
    else:
        df, origin = fetch_series(ind["src"], ind["code"])
        if df is None:
            return ind, None, origin
    df = df[(df["year"] >= START_YEAR) & df["iso3"].isin(valid_iso3)].dropna(subset=["value"]).copy()
    df["indicator"] = ind["code"]
    df["src"] = ind["src"]
    return ind, df[["iso3", "year", "indicator", "value", "src"]], origin


def _merge_save(path, new: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        both = pd.concat([old, new], ignore_index=True)
        both = both.drop_duplicates(["iso3", "year", "indicator"], keep="last")
    else:
        both = new
    both = both.sort_values(["indicator", "iso3", "year"])
    both.to_csv(path, index=False)
    return both


def run_agent(agent_id: str, valid_iso3: set[str], log=print, workers: int = 6) -> dict:
    if agent_id == "_aux":
        inds, name = AUX_INDICATORS, "Indicateurs auxiliaires"
    else:
        agent = next(a for a in AGENTS if a["id"] == agent_id)
        inds, name = agent["indicateurs"], agent["nom"]
    now = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    frames, status = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_indicator, ind, valid_iso3) for ind in inds]
        for f in as_completed(futs):
            ind, df, origin = f.result()
            ok = df is not None and len(df) > 0
            status.append((ind["code"], ok, origin, 0 if not ok else len(df)))
            if ok:
                frames.append(df)
    path = AGENTS_DIR / f"{agent_id}.csv"
    if frames:
        data = _merge_save(path, pd.concat(frames, ignore_index=True))
    else:
        data = pd.read_csv(path) if path.exists() else pd.DataFrame()
    n_ok = sum(s[1] for s in status) + sum(1 for s in status if s[2] == "conservé")
    log(f"Agent {name} : {n_ok}/{len(inds)} séries mises à jour, {len(data)} observations en historique")
    row = dict(run_at=now, agent=agent_id, series_ok=n_ok, series_total=len(inds),
               rows=len(data), countries=data["iso3"].nunique() if len(data) else 0,
               failed=";".join(s[0] for s in status if not s[1] and s[2] != "conservé"),
               kept=";".join(s[0] for s in status if s[2] == "conservé"),
               origins=";".join(sorted({s[2] for s in status if s[1]})))
    pd.DataFrame([row]).to_csv(RUN_LOG, mode="a", header=not RUN_LOG.exists(), index=False)
    return row


def run_all(valid_iso3: set[str], log=print) -> list[dict]:
    out = []
    for a in AGENTS:
        out.append(run_agent(a["id"], valid_iso3, log=log))
    out.append(run_agent("_aux", valid_iso3, log=log))
    return out


def load_all_long() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(AGENTS_DIR.glob("*.csv")) if not p.name.startswith("_runs")]
    if not frames:
        return pd.DataFrame(columns=["iso3", "year", "indicator", "value", "src"])
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["iso3", "year", "indicator"], keep="last")
