"""Orchestrateur : lance les agents, met à jour les CSV, réentraîne le ML, recalcule les scores.

Usage :
    python -m core.pipeline                 # tout
    python -m core.pipeline --skip-fetch    # recalcul seul à partir des CSV existants
    python -m core.pipeline --with-markets --with-calendar
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

import pandas as pd

from .agents import load_all_long, run_agent
from .catalog import AGENTS
from .countries import build_reference, load_countries
from .elections import political_risk_index, refresh_calendar_from_wikipedia
from .ml_engine import run_ml
from .paths import SCORES
from .scoring import compute_scores
from .store import build_master


def run(fetch: bool = True, markets: bool = False, calendar: bool = False, log=print, progress=None) -> dict:
    t0 = time.time()
    steps = (len(AGENTS) + 1 if fetch else 0) + 4
    done = 0

    def tick(msg):
        nonlocal done
        done += 1
        log(msg)
        if progress:
            progress(min(done / steps, 1.0), msg)

    if fetch:
        try:
            countries = build_reference()
        except Exception as e:
            log(f"Référentiel pays non actualisé ({e}) — version locale utilisée")
            countries = load_countries()
        iso = set(countries["iso3"])
        for a in AGENTS:
            r = run_agent(a["id"], iso, log=lambda m: None)
            tick(f"{a['icone']} Agent {a['nom']} : {r['series_ok']}/{r['series_total']} séries, {r['countries']} pays")
        r = run_agent("_aux", iso, log=lambda m: None)
        tick(f"🔭 Agent Démographie & auxiliaires : {r['series_ok']}/{r['series_total']} séries")
    else:
        countries = load_countries()
    long = load_all_long()
    scores = compute_scores(long, countries)
    tick(f"🧠 Agent Synthèse : {int(scores['current'].notna().sum())} pays et territoires notés")
    ml = run_ml(long, scores, log=lambda m: None)
    tick(f"🤖 Machine learning réentraîné (AUC risque politique = {ml['metrics']['auc_test']:.2f})")
    risk_idx = political_risk_index(scores["pillars"], ml["risk"], long)
    master = build_master(scores, ml, risk_idx, countries)
    snap = master[["score_actuel", "indice_avenir", "score_global", "risque_politique"]].copy()
    snap["run_at"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    path = SCORES / "scores_snapshots.csv"
    snap.reset_index().round(2).to_csv(path, mode="a", header=not path.exists(), index=False)
    tick("🗺️ Carte mondiale et scores recalculés")
    if calendar:
        try:
            n = refresh_calendar_from_wikipedia(countries, log=log)
            log(f"🗳️ Calendrier électoral : {n} élection(s) ajoutée(s)")
        except Exception as e:
            log(f"Calendrier électoral non actualisé : {e}")
    if markets:
        from .markets import refresh_all
        try:
            refresh_all(log=log)
        except Exception as e:
            log(f"Marchés non actualisés : {e}")
    tick(f"✅ Terminé en {time.time() - t0:.0f} s")
    return dict(master=master, metrics=ml["metrics"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--with-markets", action="store_true")
    ap.add_argument("--with-calendar", action="store_true")
    a = ap.parse_args()
    run(fetch=not a.skip_fetch, markets=a.with_markets, calendar=a.with_calendar)
