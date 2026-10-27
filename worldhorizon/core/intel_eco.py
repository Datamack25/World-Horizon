"""Agent Intelligence économique — puissance, vulnérabilités, dépendances, capacité technologique.

Intégré à WorldHorizon :
 - collecte : core.sources.fetch_series("wdi", code) → miroir open-numbers puis API Banque mondiale,
   stockée dans data/intel_eco/indicators.csv (format long iso3, year, indicator, value) ;
 - normalisation identique à core/scoring.py : bornes = 5e et 95e centiles mondiaux, valeur ≤ 8 ans ;
 - pilier « Gouvernance » repris des piliers WorldHorizon (gouvernance, corruption, justice) ;
 - couleurs : COLOR_THRESHOLDS du catalogue.

Mise à jour en ligne de commande :  python -m core.intel_eco
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .catalog import AGENTS, COLOR_THRESHOLDS
from .paths import ROOT

IE_DIR = ROOT / "data" / "intel_eco"
IE_CSV = IE_DIR / "indicators.csv"
IE_LOG = IE_DIR / "_runs_log.csv"
CURRENT_YEAR = dt.date.today().year
MAX_AGE = 8
HEX = {"vert": "#2E8B57", "orange": "#E0952B", "rouge": "#C8463D", "gris": "#B7BEC6"}
LIBELLE = {"vert": "Résilient", "orange": "Sous surveillance", "rouge": "Vulnérable", "gris": "Données insuffisantes"}

# code open-numbers / WDI, libellé, pilier, format, sens (+1 résilience, -1 vulnérabilité, 0 information), transformation
IE_IND = [
    ("ny_gdp_mktp_cd", "PIB (US$ courants)", "Puissance", "usd", +1, "log"),
    ("ny_gdp_mktp_kd_zg", "Croissance du PIB (%)", "Puissance", "pct", +1, None),
    ("ny_gdp_pcap_cd", "PIB par habitant (US$)", "Puissance", "usd", +1, "log"),
    ("sl_uem_totl_zs", "Chômage (%)", "Puissance", "pct", -1, None),
    ("fp_cpi_totl_zg", "Inflation (%)", "Stabilité macro", "pct", -1, "target:2"),
    ("bn_cab_xoka_gd_zs", "Solde courant (% PIB)", "Stabilité macro", "pct", +1, None),
    ("fi_res_totl_mo", "Réserves (mois d'importations)", "Stabilité macro", "num", +1, None),
    ("gc_dod_totl_gd_zs", "Dette de l'État central (% PIB)", "Stabilité macro", "pct", -1, None),
    ("dt_dod_dect_gn_zs", "Dette extérieure (% RNB)", "Stabilité macro", "pct", -1, None),
    ("ne_trd_gnfs_zs", "Ouverture commerciale (% PIB)", "Dépendances", "pct", 0, None),
    ("bx_klt_dinv_wd_gd_zs", "IDE entrants nets (% PIB)", "Dépendances", "pct", 0, None),
    ("tx_val_fuel_zs_un", "Hydrocarbures (% exportations)", "Dépendances", "pct", -1, None),
    ("tx_val_mmtl_zs_un", "Minerais & métaux (% exportations)", "Dépendances", "pct", -1, None),
    ("eg_imp_cons_zs", "Énergie importée (% consommation)", "Dépendances", "pct", -1, None),
    ("tm_val_food_zs_un", "Alimentaire (% importations)", "Dépendances", "pct", -1, None),
    ("gb_xpd_rsdv_gd_zs", "Dépenses de R&D (% PIB)", "Technologie", "pct", +1, None),
    ("tx_val_tech_mf_zs", "Exportations high-tech (% manuf.)", "Technologie", "pct", +1, None),
    ("nv_ind_manf_zs", "Industrie manufacturière (% PIB)", "Technologie", "pct", +1, None),
    ("ip_pat_resd", "Brevets déposés (résidents)", "Technologie", "num", +1, "log"),
    ("ms_mil_xpnd_gd_zs", "Dépenses militaires (% PIB)", "Puissance", "pct", 0, None),
]
IND = {c: dict(code=c, nom=n, bloc=b, fmt=f, sens=s, transform=t) for c, n, b, f, s, t in IE_IND}
BLOCS = ["Puissance", "Stabilité macro", "Dépendances", "Technologie", "Gouvernance"]
GOV_KEYWORDS = ("gouvernance", "corruption", "justice")
MIN_COVERAGE = 0.4
MIN_BLOCS = 3

SOURCES = [
    # ---------- Doctrine & institutions françaises de l'IE ----------
    ("Doctrine FR", "SISSE – Service de l'information stratégique et de la sécurité économiques",
     "https://www.entreprises.gouv.fr/fr/sisse", "Pilote de la politique publique de sécurité économique (rattaché à la DGE). Alertes, contrôle des investissements étrangers.", "Institution"),
    ("Doctrine FR", "DGSI – Sécurité économique (flashs ingérence)",
     "https://www.dgsi.interieur.gouv.fr", "Flashs mensuels de cas réels d'ingérence économique visant des entreprises françaises.", "Institution"),
    ("Doctrine FR", "Contrôle des investissements étrangers en France (DG Trésor)",
     "https://www.tresor.economie.gouv.fr/services-aux-entreprises/investissements-etrangers-en-france", "Procédure IEF, secteurs sensibles, rapports annuels.", "Institution"),
    ("Doctrine FR", "ANSSI – Agence nationale de la sécurité des systèmes d'information",
     "https://cyber.gouv.fr", "Menace cyber, panoramas annuels, protection du patrimoine informationnel.", "Institution"),
    ("Doctrine FR", "SGDSN – Secrétariat général de la défense et de la sécurité nationale",
     "https://www.sgdsn.gouv.fr", "Revue nationale stratégique, VIGINUM (ingérences numériques étrangères).", "Institution"),
    ("Doctrine FR", "Rapport Martre (1994) – Intelligence économique et stratégie des entreprises",
     "https://www.vie-publique.fr", "Texte fondateur de l'IE en France (Commissariat général du Plan).", "Fondamental"),
    ("Doctrine FR", "Rapport Carayon (2003) – Intelligence économique, compétitivité et cohésion sociale",
     "https://www.vie-publique.fr", "Fait de l'IE une politique publique ; notion de patriotisme économique.", "Fondamental"),
    ("Doctrine FR", "Loi de blocage (1968, réformée) & extraterritorialité – rapport Gauvain (2019)",
     "https://www.vie-publique.fr", "Protection des entreprises face aux lois extraterritoriales (FCPA, sanctions US).", "Fondamental"),

    # ---------- Écoles, centres de recherche, revues ----------
    ("Recherche", "École de Guerre Économique (EGE)",
     "https://www.ege.fr", "Christian Harbulot ; rapports d'alerte, notion de guerre économique et de guerre de l'information.", "Think tank"),
    ("Recherche", "Portail de l'IE",
     "https://www.portail-ie.fr", "Veille et analyses d'IE (rattaché à l'EGE) – très bon pour l'actualité sectorielle.", "Veille"),
    ("Recherche", "IFRI – Institut français des relations internationales",
     "https://www.ifri.org", "Géoéconomie, énergie, Chine, Afrique ; notes de référence.", "Think tank"),
    ("Recherche", "IRIS – Institut de relations internationales et stratégiques",
     "https://www.iris-france.org", "Géopolitique, matières premières, défense.", "Think tank"),
    ("Recherche", "CEPII – Centre d'études prospectives et d'informations internationales",
     "http://www.cepii.fr", "Bases BACI (commerce bilatéral détaillé) et CHELEM ; panorama de l'économie mondiale.", "Données"),
    ("Recherche", "FRS – Fondation pour la recherche stratégique",
     "https://www.frstrategie.org", "Défense, technologies critiques, prolifération.", "Think tank"),
    ("Recherche", "Bruegel",
     "https://www.bruegel.org", "Think tank européen : sécurité économique UE, dépendances, politique industrielle.", "Think tank"),
    ("Recherche", "Chatham House",
     "https://www.chathamhouse.org", "Géoéconomie, ressources (resourcetrade.earth).", "Think tank"),
    ("Recherche", "CSIS – Center for Strategic and International Studies",
     "https://www.csis.org", "Technologies critiques, semi-conducteurs, contrôle des exportations US.", "Think tank"),
    ("Recherche", "Lowy Institute – Asia Power Index",
     "https://power.lowyinstitute.org", "Mesure multidimensionnelle de la puissance en Asie-Pacifique.", "Indice"),
    ("Recherche", "Revue « Sécurité globale » / « Hérodote » / « Revue Défense Nationale »",
     "https://www.defnat.com", "Revues académiques françaises de géopolitique et stratégie.", "Revue"),

    # ---------- Données macro & commerce ----------
    ("Données éco", "Banque mondiale – World Development Indicators",
     "https://data.worldbank.org", "Plus de 1 400 indicateurs par pays (source des données de cet onglet).", "Données"),
    ("Données éco", "FMI – World Economic Outlook & Article IV",
     "https://www.imf.org/en/Publications/WEO", "Projections macro ; les rapports Article IV sont la meilleure lecture pays.", "Données"),
    ("Données éco", "OCDE – Data Explorer & Études économiques par pays",
     "https://data-explorer.oecd.org", "Données comparatives, TiVA (valeur ajoutée dans les échanges).", "Données"),
    ("Données éco", "UN Comtrade",
     "https://comtradeplus.un.org", "Commerce bilatéral par produit (SH) : repérer les dépendances fines.", "Données"),
    ("Données éco", "Observatory of Economic Complexity (OEC)",
     "https://oec.world", "Visualisation des exportations/importations par produit et partenaire.", "Données"),
    ("Données éco", "Harvard – Atlas of Economic Complexity",
     "https://atlas.hks.harvard.edu", "Complexité économique, espace-produit, diversification possible.", "Données"),
    ("Données éco", "CNUCED / UNCTAD – World Investment Report",
     "https://unctad.org", "Flux d'IDE, politiques d'investissement, filtrage des investissements.", "Données"),
    ("Données éco", "WITS – World Integrated Trade Solution",
     "https://wits.worldbank.org", "Tarifs douaniers, mesures non tarifaires.", "Données"),
    ("Données éco", "Global Trade Alert",
     "https://www.globaltradealert.org", "Recensement des mesures protectionnistes et subventions dans le monde.", "Veille"),
    ("Données éco", "DG Trésor – Services économiques à l'étranger (fiches pays)",
     "https://www.tresor.economie.gouv.fr/Pays", "Analyse pays rédigée par le réseau diplomatique économique français.", "Institution"),
    ("Données éco", "Banque de France – Balance des paiements & position extérieure",
     "https://www.banque-france.fr", "Données sur la France et ses relations financières extérieures.", "Données"),

    # ---------- Risque pays & crédit ----------
    ("Risque pays", "Coface – Évaluations risque pays",
     "https://www.coface.com/fr/actualites-economie-conseils-d-experts/tableau-de-bord-des-risques-pays",
     "Notes pays et sectorielles, mises à jour trimestrielles.", "Risque"),
    ("Risque pays", "Allianz Trade – Country Risk",
     "https://www.allianz-trade.com/en_global/economic-research/country-reports.html", "Rapports pays et risque de défaut des entreprises.", "Risque"),
    ("Risque pays", "OCDE – Classification du risque pays (crédit export)",
     "https://www.oecd.org/en/topics/export-credits.html", "Échelle 0-7 utilisée par les agences de crédit export.", "Risque"),
    ("Risque pays", "Bpifrance Assurance Export",
     "https://www.bpifrance.fr", "Politique d'assurance-crédit par pays.", "Risque"),
    ("Risque pays", "Banque mondiale – Worldwide Governance Indicators",
     "https://www.worldbank.org/en/publication/worldwide-governance-indicators", "6 dimensions de gouvernance (utilisées dans le score de l'onglet).", "Indice"),
    ("Risque pays", "Transparency International – Indice de perception de la corruption",
     "https://www.transparency.org/en/cpi", "Classement annuel des pays.", "Indice"),
    ("Risque pays", "GAFI / FATF – Listes grise et noire",
     "https://www.fatf-gafi.org", "Juridictions à risque LCB-FT ; rapports d'évaluation mutuelle.", "Conformité"),
    ("Risque pays", "Basel AML Index",
     "https://index.baselgovernance.org", "Score de risque blanchiment par pays.", "Conformité"),

    # ---------- Sanctions & contrôle des exportations ----------
    ("Sanctions", "EU Sanctions Map",
     "https://www.sanctionsmap.eu", "Régimes de sanctions UE par pays.", "Conformité"),
    ("Sanctions", "OFAC – Sanctions Programs (US Treasury)",
     "https://ofac.treasury.gov", "Programmes américains, liste SDN, portée extraterritoriale.", "Conformité"),
    ("Sanctions", "BIS – Entity List (US Commerce)",
     "https://www.bis.gov", "Contrôle des exportations technologiques (semi-conducteurs, IA).", "Conformité"),
    ("Sanctions", "DG Trésor – Gel des avoirs (registre national)",
     "https://gels-avoirs.dgtresor.gouv.fr", "Registre français des personnes et entités visées.", "Conformité"),
    ("Sanctions", "Biens à double usage – Règlement (UE) 2021/821",
     "https://eur-lex.europa.eu", "Cadre européen du contrôle des exportations sensibles.", "Conformité"),

    # ---------- Ressources critiques & énergie ----------
    ("Ressources", "Commission européenne – Matières premières critiques (CRMA)",
     "https://single-market-economy.ec.europa.eu/sectors/raw-materials_en", "Liste UE des matières critiques et stratégiques.", "Institution"),
    ("Ressources", "USGS – Mineral Commodity Summaries",
     "https://www.usgs.gov/centers/national-minerals-information-center", "Production et réserves mondiales par minerai.", "Données"),
    ("Ressources", "AIE / IEA – Critical Minerals & World Energy Outlook",
     "https://www.iea.org", "Dépendances énergétiques et minérales, transition.", "Données"),
    ("Ressources", "Energy Institute – Statistical Review of World Energy",
     "https://www.energyinst.org/statistical-review", "Production / consommation énergétique par pays.", "Données"),
    ("Ressources", "FAO – FAOSTAT",
     "https://www.fao.org/faostat", "Sécurité et dépendance alimentaires.", "Données"),

    # ---------- Innovation & propriété intellectuelle ----------
    ("Innovation", "OMPI / WIPO – Global Innovation Index",
     "https://www.wipo.int/global_innovation_index", "Classement de l'innovation par pays.", "Indice"),
    ("Innovation", "WIPO PATENTSCOPE",
     "https://patentscope.wipo.int", "Recherche de brevets mondiale : veille technologique.", "Outil"),
    ("Innovation", "Espacenet (Office européen des brevets)",
     "https://worldwide.espacenet.com", "Plus de 150 M de documents brevets, gratuit.", "Outil"),
    ("Innovation", "INPI",
     "https://www.inpi.fr", "Propriété industrielle en France, protection des actifs immatériels.", "Institution"),
    ("Innovation", "ASPI – Critical Technology Tracker",
     "https://techtracker.aspi.org.au", "Qui mène la recherche dans 64 technologies critiques.", "Indice"),

    # ---------- Défense & sécurité ----------
    ("Sécurité", "SIPRI – Dépenses militaires & transferts d'armes",
     "https://www.sipri.org/databases", "Bases de référence mondiales.", "Données"),
    ("Sécurité", "ACLED – Données sur les conflits",
     "https://acleddata.com", "Événements de violence politique géolocalisés.", "Données"),

    # ---------- Entreprises & OSINT (sources ouvertes, légales) ----------
    ("OSINT entreprises", "Annuaire des entreprises (État français)",
     "https://annuaire-entreprises.data.gouv.fr", "Identité, dirigeants, bénéficiaires effectifs (accès encadré), finances.", "Outil"),
    ("OSINT entreprises", "Pappers",
     "https://www.pappers.fr", "Comptes, actes, liens capitalistiques des sociétés françaises.", "Outil"),
    ("OSINT entreprises", "BODACC",
     "https://www.bodacc.fr", "Annonces légales : procédures collectives, cessions.", "Outil"),
    ("OSINT entreprises", "OpenCorporates",
     "https://opencorporates.com", "Registre mondial des sociétés.", "Outil"),
    ("OSINT entreprises", "OpenSanctions",
     "https://www.opensanctions.org", "Sanctions + PPE agrégées, données ouvertes.", "Conformité"),
    ("OSINT entreprises", "ICIJ Offshore Leaks Database",
     "https://offshoreleaks.icij.org", "Panama/Pandora Papers : structures offshore.", "Outil"),
    ("OSINT entreprises", "Marchés publics – BOAMP & TED (UE)",
     "https://www.boamp.fr", "Veille concurrentielle sur les appels d'offres (TED : ted.europa.eu).", "Outil"),
    ("OSINT entreprises", "SEC EDGAR",
     "https://www.sec.gov/edgar", "Rapports 10-K/20-F : risques déclarés, dépendances fournisseurs.", "Outil"),

    # ---------- Ouvrages de référence ----------
    ("Ouvrages", "C. Harbulot – Manuel d'intelligence économique (PUF)",
     "https://www.puf.com", "Le manuel de référence en France.", "Livre"),
    ("Ouvrages", "A. Laïdi – Histoire mondiale de la guerre économique (Perrin)",
     "https://www.lisez.com", "Mise en perspective historique des affrontements économiques.", "Livre"),
    ("Ouvrages", "A. Laïdi – Le droit, nouvelle arme de guerre économique (Actes Sud)",
     "https://www.actes-sud.fr", "Extraterritorialité du droit américain, cas Alstom.", "Livre"),
    ("Ouvrages", "N. Moinet – Petite histoire de l'intelligence économique (L'Harmattan)",
     "https://www.editions-harmattan.fr", "Genèse et concepts de l'IE.", "Livre"),
    ("Ouvrages", "E. Luttwak – « From Geopolitics to Geo-Economics » (1990)",
     "https://www.jstor.org", "Article fondateur de la géoéconomie.", "Article"),
    ("Ouvrages", "H. Farrell & A. Newman – Underground Empire (2023)",
     "https://www.macmillan.com", "Interdépendance militarisée : dollar, SWIFT, semi-conducteurs.", "Livre"),
    ("Ouvrages", "C. Miller – Chip War (2022)",
     "https://www.simonandschuster.com", "La guerre des semi-conducteurs, cas d'école d'IE.", "Livre"),
    ("Ouvrages", "M. Porter – L'avantage concurrentiel des nations (1990)",
     "https://www.hbs.edu", "Le « diamant » de Porter, base de l'analyse de compétitivité pays.", "Livre"),
]


# =========================================================================== collecte
def update(log=print, progress=None) -> pd.DataFrame:
    """Collecte les séries IE pour tous les pays. Les séries injoignables gardent l'historique existant."""
    from .sources import fetch_series
    IE_DIR.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(IE_CSV) if IE_CSV.exists() else pd.DataFrame(columns=["iso3", "year", "indicator", "value"])
    frames, journal = [], []
    for i, (code, nom, *_rest) in enumerate(IE_IND):
        if progress:
            progress(i / len(IE_IND), f"Intelligence économique : {nom}")
        df, origine = fetch_series("wdi", code)
        if df is not None and len(df):
            df = df[df["year"] >= 1990].dropna(subset=["value"]).copy()
            df["indicator"] = code
            frames.append(df[["iso3", "year", "indicator", "value"]])
        else:
            frames.append(old[old["indicator"] == code])
            origine = "conservé (source injoignable)"
        journal.append(dict(run_at=dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"), indicator=code,
                            origine=origine, n=len(frames[-1])))
        log(f"🕵️ {nom} : {origine}")
    out = pd.concat(frames, ignore_index=True)
    out["iso3"] = out["iso3"].astype(str).str.upper()
    out.to_csv(IE_CSV, index=False)
    pd.DataFrame(journal).to_csv(IE_LOG, mode="a", header=not IE_LOG.exists(), index=False)
    if progress:
        progress(1.0, "Intelligence économique : terminé")
    return out


@st.cache_data(show_spinner=False)
def _load(_mtime: float) -> pd.DataFrame:
    return pd.read_csv(IE_CSV)


def load() -> pd.DataFrame:
    if not IE_CSV.exists():
        return pd.DataFrame(columns=["iso3", "year", "indicator", "value"])
    return _load(IE_CSV.stat().st_mtime)


# =========================================================================== scoring
def _transform(v: pd.Series, t: str | None) -> pd.Series:
    v = v.astype(float)
    if t == "log":
        return np.log10(v.clip(lower=1e-3))
    if t and t.startswith("target:"):
        return (v - float(t.split(":")[1])).abs()
    return v


def _gov_ids() -> list[str]:
    return [a["id"] for a in AGENTS if any(k in a["nom"].lower() for k in GOV_KEYWORDS)]


@st.cache_data(show_spinner=False)
def _compute(long: pd.DataFrame, iso_list: tuple, pillars: pd.DataFrame) -> dict:
    iso = list(iso_list)
    d = long[(long["year"] >= CURRENT_YEAR - MAX_AGE) & long["iso3"].isin(iso)].sort_values("year")
    last = d.groupby(["iso3", "indicator"]).last()
    values = last["value"].unstack().reindex(index=iso, columns=list(IND))
    years = last["year"].unstack().reindex(index=iso, columns=list(IND))
    scores = pd.DataFrame(index=iso, dtype=float)
    bornes = []
    for code, ind in IND.items():
        if ind["sens"] == 0 or values[code].notna().sum() <= 10:
            continue
        t = _transform(values[code], ind["transform"])
        lo, hi = np.nanpercentile(t.dropna(), [5, 95])
        if hi == lo:
            continue
        s = ((t - lo) / (hi - lo)).clip(0, 1) * 100
        scores[code] = s if ind["sens"] > 0 else 100 - s
        bornes.append(dict(code=code, indicateur=ind["nom"], borne_basse=lo, borne_haute=hi,
                           sens="↑ résilience" if ind["sens"] > 0 else "↑ vulnérabilité",
                           transformation=ind["transform"] or ""))
    blocs = pd.DataFrame(index=iso, dtype=float)
    for b in BLOCS[:-1]:
        cols = [c for c, i in IND.items() if i["bloc"] == b and i["sens"] != 0]
        dispo = [c for c in cols if c in scores]
        if not dispo:
            continue
        cov = scores[dispo].notna().sum(axis=1) / len(cols)
        blocs[b] = scores[dispo].mean(axis=1).where(cov >= MIN_COVERAGE)
    gov = [g for g in _gov_ids() if g in pillars.columns]
    blocs["Gouvernance"] = pillars.reindex(iso)[gov].mean(axis=1) if gov else np.nan
    blocs = blocs.reindex(columns=BLOCS)
    indice = blocs.mean(axis=1).where(blocs.notna().sum(axis=1) >= MIN_BLOCS)
    return dict(values=values, years=years, scores=scores, blocs=blocs, indice=indice,
                rang=indice.rank(ascending=False, method="min"), bornes=pd.DataFrame(bornes))


def compute(store) -> dict | None:
    long = load()
    if long.empty:
        return None
    return _compute(long, tuple(store.countries["iso3"]), store.pillars)


def color(score) -> str:
    if score is None or not np.isfinite(score):
        return "gris"
    return "vert" if score >= COLOR_THRESHOLDS["vert"] else "orange" if score >= COLOR_THRESHOLDS["orange"] else "rouge"


# =========================================================================== mise en forme
def _fmt(v, fmt) -> str:
    if v is None or not np.isfinite(v):
        return "n.d."
    if fmt == "usd":
        for seuil, suf in ((1e12, " Bn$"), (1e9, " Md$"), (1e6, " M$")):
            if abs(v) >= seuil:
                return f"{v / seuil:,.1f}{suf}".replace(",", " ").replace(".", ",")
        return f"{v:,.0f} $".replace(",", " ")
    if fmt == "pct":
        return f"{v:.1f} %".replace(".", ",")
    return f"{v:,.1f}".replace(",", " ").replace(".", ",")


def _kpis(items) -> str:
    cells = "".join(f'<div class="wh-kpi {c}"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l, c in items)
    return f'<div class="wh-kpis">{cells}</div>'


def _layout(fig, h):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _radar(series: dict[str, pd.Series], h=380):
    fig = go.Figure()
    pal = ["#1F5C7A", "#C8463D", "#2E8B57", "#E0952B", "#6B4C9A", "#4A5D6E"]
    for i, (nom, s) in enumerate(series.items()):
        r = [s.get(b) for b in BLOCS]
        fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=BLOCS + BLOCS[:1], name=nom, fill="toself" if i == 0 else None,
                                      line=dict(color=pal[i % len(pal)], dash="dash" if nom == "Moyenne mondiale" else None)))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], tickfont_size=9), bgcolor="rgba(0,0,0,0)"),
                      legend=dict(orientation="h"))
    return _layout(fig, h)


def alertes(v: pd.Series, blocs: pd.Series) -> list[str]:
    g = lambda c: v.get(c) if pd.notna(v.get(c)) else None  # noqa: E731
    a = []
    if (x := g("tx_val_fuel_zs_un")) is not None and x > 50:
        a.append(f"🔴 **Économie de rente** : {x:.0f} % des exportations sont des hydrocarbures — forte exposition aux prix mondiaux.")
    if (x := g("tx_val_mmtl_zs_un")) is not None and x > 40:
        a.append(f"🔴 **Dépendance minière** : {x:.0f} % des exportations — ressources convoitées par des puissances étrangères.")
    if (x := g("eg_imp_cons_zs")) is not None and x > 60:
        a.append(f"🟠 **Dépendance énergétique** : {x:.0f} % de l'énergie consommée est importée.")
    if (x := g("tm_val_food_zs_un")) is not None and x > 20:
        a.append(f"🟠 **Dépendance alimentaire** : {x:.0f} % des importations sont alimentaires.")
    if (x := g("fi_res_totl_mo")) is not None and x < 3:
        a.append(f"🔴 **Réserves de change faibles** : {x:.1f} mois d'importations (seuil d'alerte usuel du FMI : 3).")
    if (x := g("fp_cpi_totl_zg")) is not None and x > 10:
        a.append(f"🟠 **Inflation élevée** : {x:.1f} %.")
    if (x := g("bn_cab_xoka_gd_zs")) is not None and x < -5:
        a.append(f"🟠 **Déficit courant** : {x:.1f} % du PIB — dépendance aux financements extérieurs.")
    if (x := g("gc_dod_totl_gd_zs")) is not None and x > 100:
        a.append(f"🟠 **Dette publique** : {x:.0f} % du PIB.")
    if (x := g("dt_dod_dect_gn_zs")) is not None and x > 60:
        a.append(f"🔴 **Endettement extérieur** : {x:.0f} % du RNB — levier pour les créanciers étrangers.")
    if (x := g("gb_xpd_rsdv_gd_zs")) is not None and x < 0.5:
        a.append(f"🟠 **Faible effort de R&D** ({x:.2f} % du PIB) — dépendance technologique probable.")
    if (x := g("bx_klt_dinv_wd_gd_zs")) is not None and x > 8:
        a.append(f"🟠 **Poids des IDE** ({x:.1f} % du PIB) — vérifier qui contrôle les actifs stratégiques.")
    if pd.notna(blocs.get("Gouvernance")) and blocs["Gouvernance"] < COLOR_THRESHOLDS["orange"]:
        a.append(f"🔴 **Gouvernance fragile** (pilier {blocs['Gouvernance']:.0f}/100) — risques de captation, d'ingérence et LCB-FT.")
    return a


def note_synthese(nom: str, res: dict, iso: str, al: list[str]) -> str:
    b = res["blocs"].loc[iso]; v = res["values"].loc[iso]; y = res["years"].loc[iso]
    s = res["indice"].get(iso)
    lignes = [f"# Note d'intelligence économique — {nom}",
              f"_WorldHorizon · {dt.date.today():%d/%m/%Y} · Banque mondiale (WDI) et piliers WorldHorizon_", "",
              f"## Appréciation : {LIBELLE[color(s)]} — indice {_fmt(s, 'num')}/100"
              + (f" (rang {int(res['rang'][iso])}/{int(res['indice'].notna().sum())})" if pd.notna(res["rang"].get(iso)) else ""),
              "", "| Pilier | Score |", "|---|---|", *[f"| {k} | {_fmt(x, 'num')} |" for k, x in b.items()],
              "", "## Signaux d'alerte", *(al or ["Aucun seuil franchi."]),
              "", "## Indicateurs", "| Indicateur | Valeur | Année |", "|---|---|---|",
              *[f"| {i['nom']} | {_fmt(v.get(c), i['fmt'])} | {int(y[c]) if pd.notna(y.get(c)) else 'n.d.'} |" for c, i in IND.items()],
              "", "## Pistes à approfondir",
              "- Concentration des partenaires et des produits (OEC, UN Comtrade)",
              "- Contrôle des actifs stratégiques par des capitaux étrangers",
              "- Exposition aux sanctions et aux lois extraterritoriales",
              "- Dernier rapport Article IV du FMI, note Coface / Allianz Trade"]
    return "\n".join(lignes)


# =========================================================================== panneaux
def _no_data():
    st.info("Les données d'intelligence économique n'ont pas encore été collectées.")
    update_button("first")


def update_button(key: str):
    from . import github_sync
    push = github_sync.configured()
    if st.button("🔄 Collecter / mettre à jour les données d'intelligence économique", key=f"ieupd_{key}", type="primary"):
        with st.status("Collecte des indicateurs…", expanded=True) as s:
            bar = st.progress(0.0)
            try:
                update(log=st.write, progress=lambda f, m: bar.progress(f, text=m))
                if push:
                    st.write(f"💾 Sauvegardé dans GitHub : {github_sync.commit_data()}")
                s.update(label="Données à jour", state="complete")
            except Exception as e:  # noqa: BLE001
                s.update(label=f"Collecte interrompue : {e}", state="error")
                return
        st.cache_data.clear()
        st.rerun()


def country_panel(store, iso: str, res: dict | None = None, key: str = "p"):
    """Analyse IE d'un pays : réutilisée dans la page dédiée et dans l'onglet de « Analyse d'un pays »."""
    res = res or compute(store)
    if res is None:
        _no_data(); return
    nom = store.name(iso)
    b = res["blocs"].loc[iso]; v = res["values"].loc[iso]; y = res["years"].loc[iso]
    s = res["indice"].get(iso)
    n = int(res["indice"].notna().sum())
    rang = f"{int(res['rang'][iso])}<span style='font-size:1rem'>e/{n}</span>" if pd.notna(res["rang"].get(iso)) else "n.d."
    st.markdown(_kpis([(_fmt(s, "num").split(",")[0], "Indice de résilience IE /100", color(s)), (rang, "Rang mondial", ""),
                       *[(_fmt(b[k], "num").split(",")[0], k, color(b[k])) for k in BLOCS]]), unsafe_allow_html=True)
    l, r = st.columns([1, 1.1])
    with l:
        st.plotly_chart(_radar({nom: b, "Moyenne mondiale": res["blocs"].mean()}), width="stretch", key=f"ierad_{key}")
    with r:
        st.markdown("##### 🚨 Signaux faibles et alertes")
        al = alertes(v, b)
        if al:
            st.markdown("\n".join(f"- {x}" for x in al))
        else:
            st.success("Aucun seuil d'alerte franchi sur les indicateurs disponibles.")
    t = pd.DataFrame([dict(Pilier=i["bloc"], Indicateur=i["nom"], Valeur=_fmt(v.get(c), i["fmt"]),
                           Année=int(y[c]) if pd.notna(y.get(c)) else None,
                           Score=round(res["scores"].loc[iso, c]) if c in res["scores"] and pd.notna(res["scores"].loc[iso, c]) else None)
                      for c, i in IND.items()])
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={"Score": st.column_config.ProgressColumn("Score /100", min_value=0, max_value=100, format="%d"),
                                "Année": st.column_config.NumberColumn(format="%d")})
    st.caption("Gouvernance = moyenne des piliers WorldHorizon « gouvernance », « corruption » et « justice ». "
               "Indicateurs « information » (ouverture, IDE, défense) : non notés, à interpréter.")
    a, c2 = st.columns(2)
    a.download_button("⬇️ Note de synthèse (.md)", note_synthese(nom, res, iso, al), f"WorldHorizon_{iso}_IE.md", width="stretch", key=f"ienote_{key}")
    c2.download_button("⬇️ Indicateurs (.csv)", t.to_csv(index=False).encode("utf-8"), f"WorldHorizon_{iso}_IE.csv", "text/csv",
                       width="stretch", key=f"iecsv_{key}")


def dependencies_panel(store, iso: str, key: str = "p"):
    long = load()
    codes = [c for c, i in IND.items() if i["bloc"] == "Dépendances"]
    sel = st.multiselect("Indicateurs", codes, default=codes[2:6], format_func=lambda c: IND[c]["nom"], key=f"iedep_{key}")
    d = long[(long.iso3 == iso) & long.indicator.isin(sel)].copy()
    if d.empty:
        st.info("Pas de série disponible pour ce pays.")
    else:
        d["Indicateur"] = d.indicator.map(lambda c: IND[c]["nom"])
        st.plotly_chart(_layout(px.line(d.sort_values("year"), x="year", y="value", color="Indicateur", markers=True,
                                         labels={"year": "", "value": "%"}), 380), width="stretch", key=f"iedepc_{key}")
    st.markdown(f"**Aller plus loin, produit par produit :** [OEC — profil du pays](https://oec.world/en/profile/country/{iso.lower()}) · "
                "[UN Comtrade](https://comtradeplus.un.org) · [Atlas of Economic Complexity](https://atlas.hks.harvard.edu) · "
                "[Matières premières critiques (UE)](https://single-market-economy.ec.europa.eu/sectors/raw-materials_en)")


def _world_map(store, res):
    df = store.countries[["iso3", "name_fr"]].copy()
    df["val"] = df["iso3"].map(res["indice"])
    df["coul"] = df["val"].map(color)
    df["txt"] = df["val"].map(lambda x: _fmt(x, "num"))
    fig = go.Figure()
    for k in ("vert", "orange", "rouge", "gris"):
        sub = df[df.coul == k]
        fig.add_trace(go.Choropleth(locations=sub.iso3, z=[1] * len(sub), locationmode="ISO-3", showscale=False,
                                    colorscale=[[0, HEX[k]], [1, HEX[k]]], customdata=np.stack([sub.name_fr, sub.txt], axis=1) if len(sub) else None,
                                    hovertemplate="<b>%{customdata[0]}</b><br>Indice IE : %{customdata[1]}<extra></extra>",
                                    marker_line_color="#fff", marker_line_width=0.4))
    fig.update_geos(projection_type="natural earth", showframe=False, showcoastlines=False, showland=True, landcolor="#DCE2DE",
                    showocean=True, oceancolor="#F3F5F2", lataxis_range=[-58, 85])
    fig.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", dragmode=False)
    return fig


# =========================================================================== page dédiée
def page(store, iso_list: list[str], labels: dict[str, str]):
    st.markdown("<div class='wh-hero'><h1 class='wh-title'>Intelligence économique</h1>"
                "<p>Veille · sécurité économique · influence. Qui est puissant, qui est vulnérable, qui dépend de qui : "
                "rente, endettement, énergie, alimentation, technologie et gouvernance, pays par pays.</p></div>", unsafe_allow_html=True)
    res = compute(store)
    tabs = st.tabs(["🌍 Carte mondiale", "🔎 Analyse pays", "⛓️ Dépendances", "⚖️ Comparaison",
                    "🧭 Grilles d'analyse", "📚 Sources", "🎓 Méthode"])
    if res is not None:
        cur = st.session_state.get("iso")
        iso = st.session_state.get("ie_iso") or (cur if cur in iso_list else ("FRA" if "FRA" in iso_list else iso_list[0]))
    with tabs[0]:
        if res is None:
            _no_data()
        else:
            ev = st.plotly_chart(_world_map(store, res), width="stretch", on_select="rerun", selection_mode="points",
                                 key="ie_map", config={"displayModeBar": False, "scrollZoom": False})
            try:
                pts = ev.selection.points
                if pts and pts[0].get("location") in iso_list:
                    st.session_state["ie_iso"] = st.session_state["iso"] = pts[0]["location"]
                    st.toast(f"{store.name(pts[0]['location'])} sélectionné — ouvrez l'onglet « Analyse pays ».")
            except Exception:  # noqa: BLE001
                pass
            t = store.countries.set_index("iso3")[["name_fr"]].join(res["blocs"].round(0))
            t["Indice"] = res["indice"].round(0)
            t = t.dropna(subset=["Indice"]).sort_values("Indice", ascending=False)
            a, b = st.columns(2)
            a.subheader("Les plus résilients"); a.dataframe(t.head(12), width="stretch", column_config={"name_fr": "Pays"})
            b.subheader("Les plus vulnérables"); b.dataframe(t.tail(12).iloc[::-1], width="stretch", column_config={"name_fr": "Pays"})
            st.download_button("⬇️ Classement complet (CSV)", t.to_csv().encode("utf-8"), "worldhorizon_intelligence_economique.csv", "text/csv")
    with tabs[1]:
        if res is not None:
            iso = st.selectbox("Pays analysé", iso_list, index=iso_list.index(iso), format_func=labels.get, key="ie_sel")
            st.session_state["ie_iso"] = iso
            country_panel(store, iso, res, key="page")
        else:
            _no_data()
    with tabs[2]:
        if res is not None:
            st.caption(f"Pays : {store.name(iso)} (changez-le dans l'onglet « Analyse pays »).")
            dependencies_panel(store, iso, key="page")
    with tabs[3]:
        if res is not None:
            default = list(dict.fromkeys([iso, "FRA", "DEU", "USA", "CHN"]))[:5]
            sel = st.multiselect("Pays (jusqu'à 6)", iso_list, default=[x for x in default if x in iso_list],
                                 format_func=labels.get, max_selections=6, key="ie_cmp")
            if sel:
                st.plotly_chart(_radar({store.name(i): res["blocs"].loc[i] for i in sel}, 460), width="stretch")
                t = res["blocs"].loc[sel].round(0)
                t.insert(0, "Indice", res["indice"].loc[sel].round(0))
                t.index = [store.name(i) for i in sel]
                st.dataframe(t, width="stretch")
                raw = res["values"].loc[sel, list(IND)].T
                raw.index = [IND[c]["nom"] for c in raw.index]
                raw.columns = [store.name(i) for i in sel]
                st.dataframe(raw.apply(lambda col: [_fmt(x, IND[c]["fmt"]) for x, c in zip(col, IND)]), width="stretch")
    with tabs[4]:
        grilles(iso if res is not None else "pays")
    with tabs[5]:
        bibliotheque()
    with tabs[6]:
        methode(res)
        if IE_LOG.exists():
            with st.expander("Journal des collectes"):
                st.dataframe(pd.read_csv(IE_LOG).iloc[::-1].head(40), hide_index=True, width="stretch")
        update_button("page")


# =========================================================================== grilles, sources, méthode
def grilles(iso: str):
    st.markdown("Grilles classiques de l'analyste. Les saisies sont conservées pendant la session et exportables.")
    modeles = {
        "PESTEL": ["Politique", "Économique", "Socioculturel", "Technologique", "Environnemental", "Légal"],
        "SWOT géoéconomique": ["Forces", "Faiblesses", "Opportunités", "Menaces"],
        "Diamant de Porter": ["Conditions des facteurs", "Conditions de la demande", "Industries amont et apparentées",
                              "Stratégie, structure et rivalité", "Rôle de l'État", "Hasard / chocs"],
        "Cartographie des acteurs": ["Acteurs étatiques", "Entreprises dominantes", "Puissances étrangères",
                                     "ONG / société civile", "Médias et relais d'opinion", "Réseaux informels"],
    }
    aides = {"Politique": "Stabilité, alliances, sanctions subies", "Économique": "Rente, dette, devise, dépendances",
             "Technologique": "R&D, brevets, technologies critiques", "Légal": "Extraterritorialité, protection des investissements",
             "Menaces": "Ingérences, prédation d'actifs, cyberattaques, sanctions"}
    g = st.radio("Grille", list(modeles), horizontal=True, key="ie_grille")
    vals, cols = {}, st.columns(2)
    for i, rub in enumerate(modeles[g]):
        vals[rub] = cols[i % 2].text_area(rub, key=f"ie_{g}_{iso}_{rub}", height=110, placeholder=aides.get(rub, ""))
    out = f"# {g} — {iso}\n\n" + "\n\n".join(f"## {k}\n{v or '_à compléter_'}" for k, v in vals.items())
    st.download_button("⬇️ Exporter la grille (.md)", out, f"{g}_{iso}.md".replace(" ", "_"))


def bibliotheque():
    df = pd.DataFrame(SOURCES, columns=["Catégorie", "Source", "Lien", "Intérêt", "Type"])
    st.markdown(f"**{len(df)} sources de référence** : institutions, données, think tanks, outils OSINT légaux, ouvrages fondateurs.")
    c1, c2, c3 = st.columns([2, 2, 3])
    cat = c1.multiselect("Catégorie", sorted(df["Catégorie"].unique()), key="ie_src_cat")
    typ = c2.multiselect("Type", sorted(df["Type"].unique()), key="ie_src_type")
    q = c3.text_input("Recherche", key="ie_src_q", placeholder="ex. sanctions, brevets, Chine…")
    f = df
    if cat: f = f[f["Catégorie"].isin(cat)]
    if typ: f = f[f["Type"].isin(typ)]
    if q: f = f[f.apply(lambda r: q.lower() in " ".join(r.astype(str)).lower(), axis=1)]
    st.dataframe(f, hide_index=True, width="stretch", column_config={"Lien": st.column_config.LinkColumn("Lien", display_text="ouvrir")})
    st.download_button("⬇️ Bibliothèque complète (.csv)", df.to_csv(index=False).encode("utf-8"), "sources_intelligence_economique.csv", "text/csv")
    with st.expander("🎯 Parcours de lecture pour devenir très solide"):
        st.markdown(
            "1. **Fondamentaux** : rapports Martre (1994) et Carayon (2003), *Manuel d'intelligence économique* (Harbulot).\n"
            "2. **Géoéconomie** : Luttwak (1990), *Histoire mondiale de la guerre économique* (Laïdi), Porter pour la compétitivité des nations.\n"
            "3. **Armes contemporaines** : *Le droit, nouvelle arme de guerre économique* (Laïdi), *Underground Empire* (Farrell & Newman), *Chip War* (Miller).\n"
            "4. **Pratique** : flashs ingérence de la DGSI, publications du SISSE, Portail de l'IE, notes de l'IFRI.\n"
            "5. **Données** : WDI, Comtrade/OEC, Article IV du FMI, notes Coface — c'est ce qui distingue l'analyste du commentateur.")


def methode(res: dict | None):
    st.markdown(f"""
#### Les trois piliers de l'intelligence économique
| Pilier | Objet | Questions types |
|---|---|---|
| **Veille / anticipation** | Collecter et analyser l'information ouverte | Qui produit quoi ? Qui dépend de qui ? Quels signaux faibles ? |
| **Sécurité économique** | Protéger le patrimoine matériel et immatériel | Quels actifs sont convoités ? Par qui ? Par quels moyens ? |
| **Influence** | Peser sur les normes, les décisions et les perceptions | Qui écrit les normes ? Quels relais ? Quels récits ? |

#### Le cycle du renseignement
1. **Besoin** : quelle décision l'analyse doit-elle éclairer ?
2. **Collecte** : sources ouvertes et légales uniquement.
3. **Traitement** : vérification, recoupement (deux sources indépendantes au moins), cotation.
4. **Analyse** : grilles PESTEL, SWOT, Porter, acteurs ; scénarios.
5. **Diffusion** : note courte, orientée décision.
6. **Retour** : le besoin est-il couvert ?

#### Cotation des sources (grille « Amiral » OTAN)
Fiabilité de la source de **A** (sûre) à **F** (non évaluable) ; crédibilité de l'information de **1** (confirmée) à **6** (non évaluable).
Une information « B2 » vient d'une source habituellement fiable et est probablement vraie.

#### Indice de résilience IE de WorldHorizon
- Même méthode que les 14 agents : dernière valeur connue (≤ {MAX_AGE} ans), bornes = 5e et 95e centiles mondiaux, score 0-100.
- Les indicateurs dont un niveau élevé fragilise le pays (dette, rente, énergie importée…) sont inversés ; l'inflation est mesurée en écart à 2 %.
- Pilier = moyenne de ses indicateurs (couverture ≥ {int(MIN_COVERAGE * 100)} %) ; Gouvernance = piliers WorldHorizon gouvernance, corruption, justice.
- Indice = moyenne des piliers (au moins {MIN_BLOCS} sur 5). Vert ≥ {COLOR_THRESHOLDS['vert']}, orange ≥ {COLOR_THRESHOLDS['orange']}, rouge en dessous.
- **Limites** : retards de publication (1 à 3 ans), pays peu couverts, vulnérabilités structurelles et non prévision. À croiser avec Coface, le FMI et la presse.

#### Éthique et cadre légal
L'intelligence économique n'est pas l'espionnage : uniquement de l'information **ouverte et légalement accessible**,
dans le respect du RGPD et du secret des affaires (loi du 30 juillet 2018).
""")
    if res is not None and len(res["bornes"]):
        with st.expander("Bornes de normalisation calculées"):
            st.dataframe(res["bornes"].round(3), hide_index=True, width="stretch")


if __name__ == "__main__":
    update()
