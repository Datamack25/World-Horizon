"""Agent Veille actualités & déclarations politiques.

Sources (sans clé API) :
  - Google News RSS (fr) : titres récents par pays ;
  - GDELT DOC 2.0 : tonalité médiatique moyenne sur 3 mois.
Chaque collecte est ajoutée à data/news/<ISO3>.csv (historique conservé).
Analyse : lexique de tonalité FR/EN, repérage des déclarations politiques et des signaux
d'impunité (scandale, non-lieu, relaxe, corruption…), sans modèle externe.
"""
from __future__ import annotations

import datetime as dt
import re
import urllib.parse
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from .paths import NEWS

POS = ("accord", "croissance", "hausse des investissements", "record", "paix", "réforme", "succès", "progrès",
       "relance", "victoire", "investit", "améliore", "stabilité", "agreement", "growth", "peace", "boost", "improve")
NEG = ("crise", "attentat", "guerre", "conflit", "violence", "mort", "tués", "corruption", "scandale", "manifestation",
       "grève", "récession", "inflation", "sanction", "coup d'état", "émeute", "fraude", "arrestation", "chômage",
       "pénurie", "catastrophe", "inondation", "séisme", "crisis", "war", "attack", "killed", "protest", "fraud")
STATEMENT = ("déclare", "annonce", "promet", "affirme", "dénonce", "appelle", "président", "premier ministre",
             "ministre", "gouvernement", "discours", "said", "says", "announces", "vows", "president", "minister")
IMPUNITY = ("impunité", "non-lieu", "relaxé", "relaxe", "acquitté", "classée sans suite", "prescription",
            "corruption", "détournement", "blanchiment", "pot-de-vin", "scandale", "immunité", "grâce présidentielle",
            "impunity", "acquitted", "bribery", "embezzlement", "money laundering")


def _tone(t: str) -> int:
    t = t.lower()
    return sum(w in t for w in POS) - sum(w in t for w in NEG)


def google_news(query: str, lang: str = "fr", max_items: int = 30) -> pd.DataFrame:
    url = (f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}"
           f"&hl={lang}&gl=FR&ceid=FR:{lang}")
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 WorldHorizon"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    rows = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        rows.append(dict(titre=title, lien=it.findtext("link"), date=it.findtext("pubDate"),
                         media=(it.find("source").text if it.find("source") is not None else "")))
        if len(rows) >= max_items:
            break
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return df


def gdelt_tone(name_en: str) -> float | None:
    q = urllib.parse.quote(f'"{name_en}" sourcelang:english')
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=tonechart&format=json&timespan=3months"
    try:
        js = requests.get(url, timeout=20).json()
        bins = js.get("tonechart", [])
        tot = sum(b["count"] for b in bins)
        return sum(b["bin"] * b["count"] for b in bins) / tot if tot else None
    except Exception:
        return None


def analyse(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return dict(n=0)
    d = df.copy()
    low = d["titre"].str.lower()
    d["tonalite"] = d["titre"].map(_tone)
    d["declaration"] = low.map(lambda t: any(w in t for w in STATEMENT))
    d["signal_impunite"] = low.map(lambda t: any(w in t for w in IMPUNITY))
    return dict(n=len(d), tonalite_moyenne=float(d["tonalite"].mean()),
                part_negative=float((d["tonalite"] < 0).mean()), part_positive=float((d["tonalite"] > 0).mean()),
                n_declarations=int(d["declaration"].sum()), n_signaux_impunite=int(d["signal_impunite"].sum()),
                table=d)


def collect(iso3: str, name_fr: str, name_en: str) -> dict:
    out = dict(iso3=iso3, ok=False)
    frames = []
    for q, lang in ((name_fr, "fr"), (f"{name_fr} gouvernement OR président OR élection", "fr")):
        try:
            frames.append(google_news(q, lang))
        except Exception as e:
            out["erreur"] = str(e)
    if frames:
        df = pd.concat(frames).drop_duplicates("titre")
        df["collecte"] = dt.datetime.utcnow().strftime("%Y-%m-%d")
        path = NEWS / f"{iso3}.csv"
        if path.exists():
            old = pd.read_csv(path, parse_dates=["date"])
            df = pd.concat([old, df]).drop_duplicates("titre", keep="last")
        df = df.sort_values("date", ascending=False)
        df.to_csv(path, index=False)
        out.update(ok=True, **analyse(df.head(60)))
    out["gdelt_tone"] = gdelt_tone(name_en)
    return out


def load_cached(iso3: str) -> dict:
    path = NEWS / f"{iso3}.csv"
    if not path.exists():
        return dict(iso3=iso3, ok=False)
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date", ascending=False)
    return dict(iso3=iso3, ok=True, cache=True, **analyse(df.head(60)))


def wikipedia_summary(name_fr: str) -> dict | None:
    """Résumé encyclopédique (Wikipédia FR) — contexte historique et institutionnel."""
    try:
        r = requests.get(f"https://fr.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(name_fr.replace(' ', '_'))}",
                         timeout=15, headers={"User-Agent": "WorldHorizon/1.0"})
        if r.status_code == 200:
            js = r.json()
            return dict(texte=js.get("extract", ""), url=js.get("content_urls", {}).get("desktop", {}).get("page"))
    except Exception:
        return None
    return None


def clean_title(t: str) -> str:
    return re.sub(r"\s+-\s+[^-]+$", "", str(t))
