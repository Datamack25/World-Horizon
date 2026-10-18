"""Chargement en mémoire de tous les CSV produits par le pipeline (une seule fois par session)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .agents import load_all_long
from .catalog import AGENTS, GLOBAL_MIX
from .countries import load_countries
from .paths import ML_DIR, SCORES
from .scoring import color_of


def _read(path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else pd.DataFrame()


@dataclass
class Store:
    countries: pd.DataFrame
    long: pd.DataFrame
    master: pd.DataFrame
    pillars: pd.DataFrame
    coverage: pd.DataFrame
    pillars_hist: pd.DataFrame
    global_hist: pd.DataFrame
    indicators: pd.DataFrame
    goalposts: pd.DataFrame
    future: pd.DataFrame
    projections: pd.DataFrame
    proj_ind: pd.DataFrame
    risk: pd.DataFrame
    peers: pd.DataFrame
    ml_metrics: pd.DataFrame
    importance: pd.DataFrame
    names: dict = field(default_factory=dict)

    def name(self, iso3: str) -> str:
        return self.names.get(iso3, iso3)

    def country(self, iso3: str) -> pd.Series:
        return self.countries.set_index("iso3").loc[iso3]

    def series(self, iso3: str, code: str) -> pd.Series:
        d = self.long[(self.long.iso3 == iso3) & (self.long.indicator == code)]
        return d.set_index("year")["value"].sort_index()


def build_master(scores: dict, ml: dict, risk_idx: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    m = countries.set_index("iso3")[["name_fr", "name_en", "region_fr", "continent_fr", "income_fr", "territoire"]].copy()
    m["score_actuel"] = scores["current"]
    m["fiabilite"] = scores["reliability"]
    m["indice_avenir"] = ml["future"]["indice_avenir"]
    g = GLOBAL_MIX["actuel"] * m["score_actuel"] + GLOBAL_MIX["avenir"] * m["indice_avenir"]
    m["score_global"] = g.where(m["indice_avenir"].notna(), m["score_actuel"])
    m["couleur"] = m["score_global"].map(color_of)
    m["rang"] = m["score_global"].rank(ascending=False, method="min")
    m["n_classes"] = int(m["score_global"].notna().sum())
    m["risque_politique"] = risk_idx["indice_risque_politique"]
    m["niveau_risque"] = risk_idx["niveau"].astype("string")
    m["proba_recul_democratique_3ans"] = risk_idx["proba_recul_democratique_3ans"]
    for c in ("score_2050_pessimiste", "score_2050_central", "score_2050_optimiste", "var_pop_2100_pct"):
        m[c] = ml["future"][c]
    for a in AGENTS:
        m[f"p_{a['id']}"] = scores["pillars"][a["id"]]
    m.index.name = "iso3"
    m.round(3).to_csv(SCORES / "country_scores_master.csv")
    return m


def load_store() -> Store:
    countries = load_countries()
    long = load_all_long()
    master = _read(SCORES / "country_scores_master.csv").set_index("iso3")
    st = Store(
        countries=countries, long=long, master=master,
        pillars=_read(SCORES / "pillars_current.csv").set_index("iso3"),
        coverage=_read(SCORES / "coverage_current.csv").set_index("iso3"),
        pillars_hist=_read(SCORES / "pillars_history.csv"),
        global_hist=_read(SCORES / "global_history.csv"),
        indicators=_read(SCORES / "indicators_current.csv"),
        goalposts=_read(SCORES / "goalposts.csv"),
        future=_read(ML_DIR / "future_index.csv").set_index("iso3"),
        projections=_read(ML_DIR / "projections_scores.csv"),
        proj_ind=_read(ML_DIR / "projections_indicators.csv"),
        risk=_read(ML_DIR / "political_risk_index.csv").set_index("iso3"),
        peers=_read(ML_DIR / "peers.csv").set_index("iso3"),
        ml_metrics=_read(ML_DIR / "metrics_log.csv"),
        importance=_read(ML_DIR / "political_feature_importance.csv"),
    )
    st.names = dict(zip(countries["iso3"], countries["name_fr"]))
    return st
