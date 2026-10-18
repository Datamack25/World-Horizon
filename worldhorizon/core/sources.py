"""Accès aux données publiques.

Ordre de priorité pour chaque série :
 1. miroir open-numbers sur GitHub (fichier CSV brut, rapide, mis à jour régulièrement) ;
 2. API officielle de la Banque mondiale (pour les séries WDI) ;
 3. dernier CSV historique déjà présent dans data/agents (aucune perte de données).
"""
from __future__ import annotations

import io
import os
import threading
import time

import pandas as pd
import requests

RAW = "https://raw.githubusercontent.com/open-numbers"
REPOS = {
    "wdi": ("ddf--world_bank--world_development_indicators", "datapoints/ddf--datapoints--{c}--by--economy--year.csv", "economy", "year"),
    "gm": ("ddf--gapminder--fasttrack", "countries_etc_datapoints/ddf--datapoints--{c}--by--country--time.csv", "country", "time"),
    "sg": ("ddf--gapminder--systema_globalis", "countries-etc-datapoints/ddf--datapoints--{c}--by--geo--time.csv", "geo", "time"),
}
HEADERS = {"User-Agent": "WorldHorizon/1.0 (+https://github.com)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def _get(url: str, timeout: int = 60, retries: int = 2) -> requests.Response | None:
    for i in range(retries + 1):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.5 * (i + 1))
    return None


def fetch_open_numbers(src: str, code: str) -> pd.DataFrame | None:
    repo, pattern, geo_col, time_col = REPOS[src]
    url = f"{RAW}/{repo}/master/{pattern.format(c=code)}"
    r = _get(url)
    if r is None:
        return None
    df = pd.read_csv(io.StringIO(r.text))
    if code not in df.columns:
        return None
    df = df.rename(columns={geo_col: "iso3", time_col: "year", code: "value"})
    df["iso3"] = df["iso3"].astype(str).str.upper()
    return df[["iso3", "year", "value"]]


def wb_code(code: str) -> str:
    return code.upper().replace("_", ".")


def fetch_worldbank_api(code: str, start: int = 1960) -> pd.DataFrame | None:
    url = (f"https://api.worldbank.org/v2/country/all/indicator/{wb_code(code)}"
           f"?format=json&per_page=20000&date={start}:2030")
    r = _get(url, timeout=90)
    if r is None:
        return None
    try:
        js = r.json()
        rows = [(d["countryiso3code"], int(d["date"]), d["value"]) for d in js[1] if d["value"] is not None]
    except Exception:
        return None
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["iso3", "year", "value"])


VDEM_URL = "https://raw.githubusercontent.com/vdeminstitute/vdemdata/master/data/vdem.RData"
_VDEM: dict = {}
_VDEM_LOCK = threading.Lock()


def _vdem_table(codes: list[str]) -> pd.DataFrame | None:
    """Télécharge le jeu V-Dem complet (~34 Mo, ~1 Go en mémoire) et n'en garde que les colonnes utiles.
    Désactivable via WH_SKIP_VDEM=1 (ex. Streamlit Community Cloud, limité en mémoire) :
    les séries V-Dem déjà présentes dans data/agents sont alors conservées telles quelles."""
    if os.environ.get("WH_SKIP_VDEM") == "1":
        return None
    with _VDEM_LOCK:
        if "df" in _VDEM:
            return _VDEM["df"]
        try:
            import pyreadr
        except ImportError:
            return None
        from .paths import CACHE
        path = CACHE / "vdem.RData"
        r = _get(VDEM_URL, timeout=300)
        if r is not None:
            path.write_bytes(r.content)
        if not path.exists():
            return None
        res = pyreadr.read_r(str(path))
        df = next(iter(res.values()))
        keep = ["country_text_id", "year"] + [c for c in codes if c in df.columns]
        df = df[keep].rename(columns={"country_text_id": "iso3"}).copy()
        del res
        df["year"] = df["year"].astype(int)
        _VDEM["df"] = df
        try:
            path.unlink()  # le fichier brut n'est pas versionné
        except OSError:
            pass
        return df


def fetch_series(src: str, code: str) -> tuple[pd.DataFrame | None, str]:
    """Renvoie (données, origine) pour une série."""
    if src == "vdem":
        from .catalog import all_indicators
        codes = [c for (s_, c) in all_indicators() if s_ == "vdem"]
        if os.environ.get("WH_SKIP_VDEM") == "1":
            return None, "conservé"
        t = _vdem_table(codes)
        if t is None or code not in t.columns:
            return None, "indisponible"
        return t[["iso3", "year", code]].rename(columns={code: "value"}).dropna(), "github:vdeminstitute/vdemdata (V-Dem v16)"
    if src in REPOS:
        df = fetch_open_numbers(src, code)
        if df is not None and len(df):
            return df, f"github:open-numbers/{REPOS[src][0]}"
        if src == "wdi":
            df = fetch_worldbank_api(code)
            if df is not None and len(df):
                return df, "api.worldbank.org"
    return None, "indisponible"


def fetch_ref_table(repo: str, path: str) -> pd.DataFrame | None:
    r = _get(f"{RAW}/{repo}/master/{path}")
    return None if r is None else pd.read_csv(io.StringIO(r.text))


def fetch_url_csv(url: str) -> pd.DataFrame | None:
    r = _get(url)
    return None if r is None else pd.read_csv(io.StringIO(r.text))
