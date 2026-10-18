"""Référentiel pays & territoires (noms FR/EN, régions, coordonnées) et recherche."""
from __future__ import annotations

import unicodedata

import pandas as pd

from .paths import REF
from .sources import fetch_ref_table, fetch_url_csv

FR_OVERRIDES = {
    "GBR": "Royaume-Uni", "USA": "États-Unis", "RUS": "Russie", "KOR": "Corée du Sud",
    "PRK": "Corée du Nord", "IRN": "Iran", "VEN": "Venezuela", "BOL": "Bolivie", "TZA": "Tanzanie",
    "SYR": "Syrie", "LAO": "Laos", "VNM": "Viêt Nam", "MDA": "Moldavie", "CZE": "Tchéquie",
    "COD": "RD Congo", "COG": "Congo", "TWN": "Taïwan", "HKG": "Hong Kong", "MAC": "Macao",
    "PSE": "Palestine", "XKX": "Kosovo", "FSM": "Micronésie", "MKD": "Macédoine du Nord",
    "CIV": "Côte d'Ivoire", "CPV": "Cap-Vert", "SWZ": "Eswatini", "TUR": "Turquie",
    "NLD": "Pays-Bas", "BHS": "Bahamas", "GMB": "Gambie", "VCT": "Saint-Vincent-et-les-Grenadines",
    "KNA": "Saint-Kitts-et-Nevis", "LCA": "Sainte-Lucie", "VGB": "Îles Vierges britanniques",
    "VIR": "Îles Vierges américaines", "MAF": "Saint-Martin (partie française)",
    "SXM": "Sint Maarten", "CUW": "Curaçao", "PRI": "Porto Rico", "BRN": "Brunei",
    "DOM": "République dominicaine", "CAF": "Centrafrique", "ARE": "Émirats arabes unis",
}
REGION_FR = {
    "eas": "Asie de l'Est & Pacifique", "ecs": "Europe & Asie centrale", "eca": "Europe & Asie centrale", "lcn": "Amérique latine & Caraïbes",
    "mea": "Moyen-Orient & Afrique du Nord", "nac": "Amérique du Nord", "sas": "Asie du Sud",
    "ssf": "Afrique subsaharienne", "afe": "Afrique subsaharienne", "afw": "Afrique subsaharienne",
}
INCOME_FR = {"hic": "Revenu élevé", "umc": "Revenu intermédiaire supérieur",
             "lmc": "Revenu intermédiaire inférieur", "lic": "Faible revenu"}
CONTINENT_FR = {"AF": "Afrique", "AS": "Asie", "EU": "Europe", "NA": "Amérique du Nord",
                "SA": "Amérique du Sud", "OC": "Océanie", "AN": "Antarctique"}
GM_REGION_TO_WB = {"europe": "ecs", "asia": "eas", "africa": "ssf", "americas": "lcn"}

REF_FILE = REF / "countries.csv"


def build_reference() -> pd.DataFrame:
    wdi = fetch_ref_table("ddf--world_bank--world_development_indicators", "ddf--entities--economy--country.csv")
    gm = fetch_ref_table("ddf--gapminder--fasttrack", "ddf--entities--geo--country.csv")
    cc = fetch_url_csv("https://raw.githubusercontent.com/datasets/country-codes/main/data/country-codes.csv")
    if wdi is None or gm is None:
        raise RuntimeError("Référentiel pays indisponible (GitHub injoignable)")

    wdi = wdi[wdi["is--country"].astype(str).str.upper() == "TRUE"].copy()
    wdi["iso3"] = wdi["country"].str.upper()
    w = wdi[["iso3", "name", "region", "income_4level", "2_alpha_code", "currency_unit"]].rename(
        columns={"name": "name_en", "income_4level": "income", "2_alpha_code": "iso2", "currency_unit": "currency"})

    gm = gm[gm["is--country"].astype(str).str.upper() == "TRUE"].copy()
    gm["iso3"] = gm["country"].str.upper()
    g = gm[["iso3", "name", "latitude", "longitude", "world_4region", "un_state", "iso3166_1_alpha2"]].rename(
        columns={"name": "name_gm", "iso3166_1_alpha2": "iso2_gm"})

    df = w.merge(g, on="iso3", how="outer")
    df["name_en"] = df["name_en"].fillna(df["name_gm"])
    df["iso2"] = df["iso2"].fillna(df["iso2_gm"].astype("string").str.upper())
    df["region"] = df["region"].fillna(df["world_4region"].map(GM_REGION_TO_WB))
    # entités Gapminder sans code ISO (Abkhazie, etc.) exclues
    df = df[df["iso3"].str.len() == 3]

    if cc is not None:
        c = cc[["ISO3166-1-Alpha-3", "official_name_fr", "Capital", "Continent", "ISO3166-1-numeric",
                "is_independent"]].rename(columns={"ISO3166-1-Alpha-3": "iso3", "Capital": "capital",
                                                   "Continent": "continent", "ISO3166-1-numeric": "isonum"})
        df = df.merge(c, on="iso3", how="left")
    else:
        df["official_name_fr"] = None; df["capital"] = None; df["continent"] = None; df["isonum"] = None; df["is_independent"] = None
    df["name_fr"] = df["iso3"].map(FR_OVERRIDES).fillna(df["official_name_fr"]).fillna(df["name_en"])
    df["region_fr"] = df["region"].map(REGION_FR).fillna("Autre")
    df["income_fr"] = df["income"].map(INCOME_FR).fillna("n.d.")
    df["continent_fr"] = df["continent"].map(CONTINENT_FR).fillna(df["region_fr"])
    df["territoire"] = ~df["is_independent"].astype(str).str.contains("Yes", na=False) & ~df["iso3"].isin(["XKX", "TWN", "PSE"])
    df.loc[df["iso3"] == "XKX", ["continent_fr"]] = "Europe"
    df.loc[df["iso3"] == "XKX", "name_fr"] = "Kosovo"
    df.loc[df["iso3"] == "TWN", "name_fr"] = "Taïwan"
    df["isonum"] = pd.to_numeric(df["isonum"], errors="coerce").astype("Int64")
    out = df[["iso3", "iso2", "isonum", "name_fr", "name_en", "region", "region_fr", "income", "income_fr",
              "continent_fr", "capital", "currency", "latitude", "longitude", "territoire"]]
    out = out.drop_duplicates("iso3").sort_values("name_fr")
    out.to_csv(REF_FILE, index=False)
    return out


def load_countries() -> pd.DataFrame:
    if not REF_FILE.exists():
        return build_reference()
    return pd.read_csv(REF_FILE, keep_default_na=False, na_values=[""])


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalnum() or ch == " ").strip()


def search(query: str, countries: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    q = _norm(query)
    if not q:
        return countries.head(0)
    keys = countries.assign(_fr=countries["name_fr"].map(_norm), _en=countries["name_en"].map(_norm),
                            _iso=countries["iso3"].str.lower())
    exact = keys[(keys._fr == q) | (keys._en == q) | (keys._iso == q)]
    starts = keys[keys._fr.str.startswith(q) | keys._en.str.startswith(q)]
    contains = keys[keys._fr.str.contains(q, regex=False) | keys._en.str.contains(q, regex=False)]
    res = pd.concat([exact, starts, contains]).drop_duplicates("iso3")
    return res.drop(columns=["_fr", "_en", "_iso"]).head(limit)


def slugs(row) -> dict:
    """Identifiants d'URL pour les sites d'experts."""
    en = str(row.get("name_en", ""))
    simple = (en.replace(",", "").replace(".", "").replace("'", "").replace("(", "").replace(")", ""))
    slug = "-".join(simple.lower().split())
    return {
        "ISO3": row["iso3"], "iso3": row["iso3"].lower(), "slug": slug, "Slug": "-".join(simple.split()),
        "slugwb": slug.replace("-", ""), "wjp": slug.replace("-", "%20"),
        "iso2l": str(row.get("iso2", "")).lower(),
        "iso3num": row["iso3"],
    }
