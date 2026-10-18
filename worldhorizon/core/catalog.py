"""
Catalogue des agents WorldHorizon.

Chaque agent thématique possède :
  - une liste d'indicateurs publics (code, source, sens de lecture, transformation),
  - un poids dans le score actuel,
  - des références d'experts (rapports de référence à consulter pour le pays).

Sources :
  wdi  -> Banque mondiale, World Development Indicators (miroir open-numbers sur GitHub,
          ou API officielle api.worldbank.org quand elle est joignable)
  gm   -> Gapminder Fast Track (agrège IDEA, EIU, Transparency International, World Happiness
          Report, ONU WPP...) — miroir open-numbers sur GitHub
  sg   -> Gapminder Systema Globalis
  pisa -> OCDE PISA (fichier data/ref/pisa.csv)
  vdem -> V-Dem Institute (Université de Göteborg), jeu complet 1789-2025, paquet R officiel sur GitHub

direction : +1 = plus haut est mieux, -1 = plus bas est mieux
transform : None | "log" | "target:<x>" (distance à une cible, plus petit = mieux)
per_capita: True -> valeur rapportée à la population (par million d'habitants) ; un nombre = autre échelle (1000 = pour 1 000 hab., 1 = par habitant)
"""

SOURCE_LABELS = {
    "wdi": "Banque mondiale — World Development Indicators",
    "gm": "Gapminder Fast Track",
    "sg": "Gapminder Systema Globalis",
    "pisa": "OCDE — PISA 2022 (Vol. I)",
    "vdem": "V-Dem Institute — Varieties of Democracy (v16)",
}

SOURCE_URLS = {
    "wdi": "https://data.worldbank.org/indicator/{code}",
    "gm": "https://www.gapminder.org/data/",
    "sg": "https://www.gapminder.org/data/",
    "pisa": "https://www.oecd.org/en/about/programmes/pisa.html",
    "vdem": "https://v-dem.net/data/the-v-dem-dataset/",
}

# Origine réelle des séries Gapminder (pour citer le producteur primaire)
GM_ORIGIN = {
    "idea": "International IDEA — Global State of Democracy Indices",
    "eiu": "Economist Intelligence Unit — Democracy Index",
    "cpi": "Transparency International — Corruption Perceptions Index",
    "whr": "World Happiness Report (Gallup World Poll)",
    "ihme": "IHME — Global Burden of Disease",
    "pop": "Nations unies — World Population Prospects (via Gapminder)",
    "lex": "Gapminder / ONU WPP — espérance de vie",
    "gini": "Banque mondiale / Gapminder — Gini",
    "journa": "Comité pour la protection des journalistes (CPJ)",
}


def I(code, label, unit, direction, src="wdi", transform=None, per_capita=False, weight=1.0, origin=None):
    return dict(code=code, label=label, unit=unit, direction=direction, src=src,
                transform=transform, per_capita=per_capita, weight=weight, origin=origin)


AGENTS = [
    dict(
        id="sante", nom="Santé", icone="🩺", poids=8,
        description="Espérance de vie, mortalité, accès aux soins, dépenses de santé, eau et assainissement.",
        indicateurs=[
            I("sp_dyn_le00_in", "Espérance de vie à la naissance", "ans", +1, weight=2),
            I("sh_dyn_mort", "Mortalité des moins de 5 ans", "‰ naissances", -1, weight=1.5),
            I("sh_sta_mmrt", "Mortalité maternelle", "pour 100 000 naissances", -1, transform="log"),
            I("sh_med_phys_zs", "Médecins", "pour 1 000 hab.", +1),
            I("sh_xpd_chex_gd_zs", "Dépenses courantes de santé", "% du PIB", +1, weight=0.5),
            I("sh_xpd_oopc_ch_zs", "Reste à charge des ménages", "% dépenses santé", -1, weight=0.7),
            I("sh_imm_meas", "Vaccination rougeole", "% enfants 12-23 mois", +1, weight=0.7),
            I("sh_h2o_smdw_zs", "Eau potable gérée en sécurité", "% population", +1),
            I("sh_sta_smss_zs", "Assainissement géré en sécurité", "% population", +1, weight=0.7),
        ],
        experts=[
            ("OMS — profil pays", "https://data.who.int/countries/{iso3num}"),
            ("IHME — Global Burden of Disease", "https://www.healthdata.org/research-analysis/health-by-location/profiles/{slug}"),
        ],
    ),
    dict(
        id="education", nom="Éducation & PISA", icone="🎓", poids=8,
        description="Scores PISA (OCDE), alphabétisation, scolarisation, pauvreté des apprentissages, dépense publique d'éducation.",
        indicateurs=[
            I("pisa_math", "PISA — mathématiques", "points", +1, src="pisa", weight=1.5),
            I("pisa_read", "PISA — compréhension de l'écrit", "points", +1, src="pisa", weight=1.5),
            I("pisa_sci", "PISA — sciences", "points", +1, src="pisa", weight=1.5),
            I("se_adt_litr_zs", "Alphabétisation des adultes", "% 15 ans et +", +1),
            I("se_sec_enrr", "Scolarisation secondaire (brute)", "%", +1),
            I("se_ter_enrr", "Scolarisation supérieure (brute)", "%", +1),
            I("se_prm_cmpt_zs", "Achèvement du primaire", "% classe d'âge", +1, weight=0.7),
            I("se_lpv_prim", "Pauvreté des apprentissages", "% enfants", -1),
            I("se_xpd_totl_gd_zs", "Dépense publique d'éducation", "% du PIB", +1, weight=0.5),
        ],
        experts=[
            ("OCDE — résultats PISA par pays", "https://www.oecd.org/en/publications/pisa-2022-results-volume-i-and-ii-country-notes_ed6fbcc5-en.html"),
            ("UNESCO UIS — données pays", "https://uis.unesco.org/en/country/{iso2l}"),
        ],
    ),
    dict(
        id="economie", nom="Économie & Macro", icone="📈", poids=9,
        description="Niveau de vie, croissance, inflation, chômage, dette, solde courant, ouverture.",
        indicateurs=[
            I("ny_gdp_pcap_pp_kd", "PIB par habitant (PPA, $ 2021 constants)", "$", +1, transform="log", weight=2),
            I("ny_gdp_mktp_kd_zg", "Croissance du PIB réel", "%/an", +1, weight=1.2),
            I("fp_cpi_totl_zg", "Inflation (écart à 2 %)", "%", -1, transform="target:2"),
            I("sl_uem_totl_zs", "Taux de chômage", "% population active", -1),
            I("gc_dod_totl_gd_zs", "Dette de l'administration centrale", "% du PIB", -1, weight=0.7),
            I("bn_cab_xoka_gd_zs", "Solde courant", "% du PIB", +1, weight=0.6),
            I("gc_tax_totl_gd_zs", "Recettes fiscales", "% du PIB", +1, weight=0.4),
            I("ne_trd_gnfs_zs", "Ouverture commerciale", "% du PIB", +1, weight=0.4),
        ],
        experts=[
            ("FMI — page pays & consultations Article IV", "https://www.imf.org/en/Countries/{ISO3}"),
            ("Banque mondiale — vue d'ensemble pays", "https://www.worldbank.org/en/country/{slugwb}/overview"),
            ("OCDE — études économiques", "https://www.oecd.org/en/countries/{slug}.html"),
        ],
    ),
    dict(
        id="finance", nom="Marchés financiers & Finance", icone="💹", poids=5,
        description="Profondeur boursière, crédit au secteur privé, taux réels, réserves de change, inclusion financière.",
        indicateurs=[
            I("cm_mkt_lcap_gd_zs", "Capitalisation boursière", "% du PIB", +1, transform="log"),
            I("cm_mkt_trad_gd_zs", "Valeur des actions échangées", "% du PIB", +1, transform="log", weight=0.6),
            I("fs_ast_prvt_gd_zs", "Crédit intérieur au secteur privé", "% du PIB", +1, weight=1.2),
            I("fr_inr_rinr", "Taux d'intérêt réel", "%", -1, transform="target:2", weight=0.6),
            I("fr_inr_lndp", "Marge d'intermédiation bancaire", "pts", -1, weight=0.6),
            I("fi_res_totl_mo", "Réserves de change", "mois d'importations", +1, weight=0.8),
            I("dt_dod_dect_gn_zs", "Dette extérieure", "% du RNB", -1, weight=0.6),
            I("fx_own_totl_zs", "Détention d'un compte (inclusion financière)", "% 15 ans et +", +1),
        ],
        experts=[
            ("FMI — Global Financial Stability Report", "https://www.imf.org/en/Publications/GFSR"),
            ("BRI — statistiques pays", "https://data.bis.org/"),
        ],
    ),
    dict(
        id="business", nom="Business & Entrepreneuriat", icone="🏢", poids=6,
        description="Création d'entreprises, IDE, accès au crédit, prévisibilité du droit des affaires (successeur de Doing Business : B-READY).",
        indicateurs=[
            I("ic_bus_ndns_zs", "Densité de créations d'entreprises", "pour 1 000 adultes", +1, transform="log", weight=1.5),
            I("bx_klt_dinv_wd_gd_zs", "Investissements directs étrangers entrants", "% du PIB", +1, weight=0.8),
            I("fr_inr_lend", "Taux débiteur bancaire", "%", -1, weight=0.6),
            I("it_net_bbnd_p2", "Haut débit fixe", "pour 100 hab.", +1, weight=0.6),
            I("eg_elc_accs_zs", "Accès à l'électricité", "% population", +1, weight=0.8),
            I("prenforct_idea", "Application prévisible des règles (IDEA)", "0-100", +1, src="gm", weight=1.2, origin="idea"),
            I("impadmin_idea", "Administration impartiale (IDEA)", "0-100", +1, src="gm", weight=0.8, origin="idea"),
        ],
        experts=[
            ("Banque mondiale — Business Ready (B-READY)", "https://www.worldbank.org/en/businessready"),
            ("Banque mondiale — Enterprise Surveys", "https://www.enterprisesurveys.org/en/data/exploreeconomies"),
        ],
    ),
    dict(
        id="environnement", nom="Environnement & Climat", icone="🌍", poids=7,
        description="Émissions, intensité carbone, renouvelables, forêts, aires protégées, stress hydrique, exposition aux catastrophes.",
        indicateurs=[
            I("en_ghg_co2_pc_ce_ar5", "CO₂ par habitant", "t CO₂e/hab.", -1),
            I("en_ghg_co2_rt_gdp_pp_kd", "Intensité carbone du PIB", "kg CO₂e / $ PPA", -1),
            I("eg_fec_rnew_zs", "Part des renouvelables", "% conso. finale", +1, weight=1.2),
            I("ag_lnd_frst_zs", "Couverture forestière", "% du territoire", +1, weight=0.6),
            I("er_ptd_totl_zs", "Aires protégées", "% du territoire", +1, weight=0.6),
            I("er_h2o_fwst_zs", "Stress hydrique", "% ressources prélevées", -1, transform="log", weight=1.2),
            I("en_clc_mdat_zs", "Exposition sécheresses/inondations/canicules", "% population", -1, weight=0.6),
        ],
        experts=[
            ("GIEC — Atlas interactif (projections régionales)", "https://interactive-atlas.ipcc.ch/"),
            ("Climate Action Tracker", "https://climateactiontracker.org/countries/"),
            ("ND-GAIN — vulnérabilité climatique", "https://gain.nd.edu/our-work/country-index/"),
        ],
    ),
    dict(
        id="agriculture", nom="Agriculture & Sécurité alimentaire", icone="🌾", poids=6,
        description="Sous-alimentation, productivité, terres arables, irrigation, poids du secteur agricole.",
        indicateurs=[
            I("sn_itk_defc_zs", "Prévalence de la sous-alimentation", "% population", -1, weight=2),
            I("ag_yld_crel_kg", "Rendement céréalier", "kg/ha", +1, transform="log"),
            I("ag_prd_food_xd", "Indice de production alimentaire", "base 2014-16 = 100", +1, weight=0.6),
            I("ag_lnd_arbl_ha_pc", "Terres arables par habitant", "ha/hab.", +1, transform="log", weight=0.6),
            I("ag_lnd_irig_ag_zs", "Terres agricoles irriguées", "%", +1, weight=0.4),
            I("ag_con_fert_zs", "Engrais utilisés", "kg/ha arable", 0, weight=0.0),
        ],
        experts=[
            ("FAO — profil pays", "https://www.fao.org/countryprofiles/index/en/?iso3={ISO3}"),
            ("Global Hunger Index", "https://www.globalhungerindex.org/"),
            ("IPC — classification de l'insécurité alimentaire", "https://www.ipcinfo.org/"),
        ],
    ),
    dict(
        id="gouvernance", nom="Gouvernance & Démocratie", icone="🏛️", poids=8,
        description="Qualité démocratique, élections libres et équitables, libertés, information pluraliste (V-Dem, EIU, IDEA).",
        indicateurs=[
            I("v2x_polyarchy", "Démocratie électorale (V-Dem)", "indice 0-1", +1, src="vdem", weight=2),
            I("v2x_libdem", "Démocratie libérale (V-Dem)", "indice 0-1", +1, src="vdem", weight=1.5),
            I("v2xel_frefair", "Élections libres et équitables", "indice 0-1", +1, src="vdem", weight=1.2),
            I("v2x_freexp_altinf", "Liberté d'expression et information alternative", "indice 0-1", +1, src="vdem"),
            I("v2x_civlib", "Libertés civiles", "indice 0-1", +1, src="vdem"),
            I("demox_eiu", "Indice de démocratie (EIU)", "0-100", +1, src="gm", weight=0.5, origin="eiu"),
            I("fppart_idea", "Liberté des partis politiques (IDEA)", "0-100", +1, src="gm", weight=0.5, origin="idea"),
            I("sg_gen_parl_zs", "Femmes au parlement", "% des sièges", +1, weight=0.5),
        ],
        experts=[
            ("Freedom House — Freedom in the World", "https://freedomhouse.org/country/{slug}/freedom-world/2025"),
            ("V-Dem — graphiques pays", "https://v-dem.net/data_analysis/CountryGraph/"),
            ("International IDEA — Democracy Tracker", "https://www.idea.int/democracytracker/country/{slug}"),
        ],
    ),
    dict(
        id="impunite", nom="Justice & Sentiment d'impunité", icone="⚖️", poids=8,
        description="État de droit, égalité devant la loi, contrôle judiciaire de l'exécutif, respect des décisions de justice, accès à la justice, homicides : mesure de l'impunité réelle et perçue.",
        indicateurs=[
            I("v2x_rule", "État de droit (V-Dem)", "indice 0-1", +1, src="vdem", weight=1.5),
            I("v2xcl_rol", "Égalité devant la loi et libertés individuelles", "indice 0-1", +1, src="vdem", weight=1.2),
            I("v2x_jucon", "Contrôle judiciaire de l'exécutif", "indice 0-1", +1, src="vdem", weight=1.2),
            I("v2jucomp", "Respect des décisions de justice par le gouvernement", "échelle V-Dem", +1, src="vdem", weight=0.8),
            I("v2exrescon", "Respect de la Constitution par l'exécutif", "échelle V-Dem", +1, src="vdem", weight=0.8),
            I("ajustice_idea", "Accès à la justice (IDEA)", "0-100", +1, src="gm", weight=0.6, origin="idea"),
            I("judicindep_idea", "Indépendance judiciaire (IDEA)", "0-100", +1, src="gm", weight=0.5, origin="idea"),
            I("vc_ihr_psrc_p5", "Homicides volontaires", "pour 100 000 hab.", -1, transform="log", weight=1),
            I("journaprison", "Journalistes emprisonnés", "nombre", -1, src="gm", transform="log", weight=0.4, origin="journa"),
        ],
        experts=[
            ("World Justice Project — Rule of Law Index", "https://worldjusticeproject.org/rule-of-law-index/global/2025"),
            ("ONUDC — données criminalité et justice", "https://dataunodc.un.org/"),
            ("Global Impunity Index (UDLAP)", "https://www.udlap.mx/igi/"),
        ],
    ),
    dict(
        id="corruption", nom="Corruption & Intégrité", icone="🕵️", poids=7,
        description="Perception de la corruption (Transparency International), corruption politique, de l'exécutif et du secteur public (V-Dem), transparence.",
        indicateurs=[
            I("corruption_perception_index_cpi", "Indice de perception de la corruption", "0-100", +1, src="gm", weight=2, origin="cpi"),
            I("v2x_corr", "Corruption politique (V-Dem)", "indice 0-1", -1, src="vdem", weight=1.5),
            I("v2x_execorr", "Corruption de l'exécutif", "indice 0-1", -1, src="vdem"),
            I("v2x_pubcorr", "Corruption du secteur public", "indice 0-1", -1, src="vdem", weight=0.8),
            I("abscorrup_idea", "Absence de corruption (IDEA)", "0-100", +1, src="gm", weight=0.5, origin="idea"),
            I("iq_cpa_tran_xq", "CPIA transparence et redevabilité", "1-6", +1, weight=0.3),
        ],
        experts=[
            ("Transparency International — page pays", "https://www.transparency.org/en/countries/{slug}"),
            ("GAFI — pays et évaluations mutuelles", "https://www.fatf-gafi.org/en/countries.html"),
            ("Basel AML Index", "https://index.baselgovernance.org/ranking"),
        ],
    ),
    dict(
        id="securite", nom="Sécurité & Paix", icone="🛡️", poids=9,
        description="Homicides, conflits armés, décès liés aux conflits et au terrorisme, militarisation, réfugiés.",
        indicateurs=[
            I("vc_ihr_psrc_p5", "Homicides volontaires", "pour 100 000 hab.", -1, transform="log", weight=1.5),
            I("vc_btl_deth", "Morts au combat", "pour 1 M hab.", -1, transform="log", per_capita=True, weight=1.5),
            I("pisec_idea", "Intégrité et sécurité des personnes (IDEA)", "0-100", +1, src="gm", weight=0.8, origin="idea"),
            I("v2x_clphy", "Protection contre la violence d'État (V-Dem)", "indice 0-1", +1, src="vdem", weight=0.8),
            I("ms_mil_xpnd_gd_zs", "Dépenses militaires", "% du PIB", -1, weight=0.4),
            I("int_refg_origin_prc", "Réfugiés originaires du pays", "% population", -1, src="gm", transform="log", weight=0.8),
        ],
        experts=[
            ("Institute for Economics & Peace — Global Peace Index", "https://www.visionofhumanity.org/maps/"),
            ("ACLED — conflits", "https://acleddata.com/"),
            ("Ministère des Affaires étrangères — Conseils aux voyageurs", "https://www.diplomatie.gouv.fr/fr/conseils-aux-voyageurs/conseils-par-pays-destination/"),
        ],
    ),
    dict(
        id="social", nom="Social & Bien-être", icone="😊", poids=8,
        description="Satisfaction de vie, inégalités, pauvreté, emploi des jeunes, participation des femmes.",
        indicateurs=[
            I("hapiscore_whr", "Score de bonheur (World Happiness Report)", "0-100", +1, src="gm", weight=2, origin="whr"),
            I("si_pov_gini", "Indice de Gini", "0-100", -1),
            I("si_pov_dday", "Pauvreté à 3 $/jour", "% population", -1, weight=1.2),
            I("sl_uem_1524_zs", "Chômage des jeunes", "% 15-24 ans", -1, weight=0.8),
            I("sl_tlf_cact_fe_zs", "Participation des femmes au marché du travail", "%", +1, weight=0.6),
            I("sh_sta_suic_p5", "Taux de suicide", "pour 100 000 hab.", -1, weight=0.4),
        ],
        experts=[
            ("World Happiness Report", "https://worldhappiness.report/"),
            ("PNUD — Rapport sur le développement humain", "https://hdr.undp.org/data-center/specific-country-data#/countries/{ISO3}"),
            ("OCDE — Better Life Index", "https://www.oecdbetterlifeindex.org/"),
        ],
    ),
    dict(
        id="innovation", nom="Innovation & Technologie", icone="🚀", poids=7,
        description="R&D, brevets, publications scientifiques, exportations high-tech, numérique.",
        indicateurs=[
            I("gb_xpd_rsdv_gd_zs", "Dépense de R&D", "% du PIB", +1, weight=1.5),
            I("ip_pat_resd", "Brevets déposés par des résidents", "par M hab.", +1, transform="log", per_capita=True, weight=1.2),
            I("ip_jrn_artc_sc", "Articles scientifiques", "par M hab.", +1, transform="log", per_capita=True),
            I("tx_val_tech_mf_zs", "Exportations high-tech", "% exports manufacturées", +1, weight=0.8),
            I("it_net_user_zs", "Utilisateurs d'Internet", "% population", +1),
            I("it_net_bbnd_p2", "Abonnements haut débit fixe", "pour 100 hab.", +1, weight=0.6),
        ],
        experts=[
            ("OMPI — Global Innovation Index", "https://www.wipo.int/web/global-innovation-index"),
            ("UIT — DataHub (numérique)", "https://datahub.itu.int/"),
            ("OCDE — AI Policy Observatory", "https://oecd.ai/en/dashboards/countries/{Slug}"),
        ],
    ),
    dict(
        id="tourisme", nom="Tourisme & Attractivité", icone="✈️", poids=4,
        description="Arrivées internationales, recettes touristiques, transferts de la diaspora, attractivité.",
        indicateurs=[
            I("st_int_arvl", "Arrivées de touristes internationaux", "pour 1 000 hab.", +1, transform="log", per_capita=1000, weight=1.5),
            I("st_int_rcpt_xp_zs", "Recettes touristiques", "% des exportations", +1),
            I("st_int_rcpt_cd", "Recettes touristiques par habitant", "$ par hab.", +1, transform="log", per_capita=1),
            I("bx_trf_pwkr_dt_gd_zs", "Transferts de la diaspora reçus", "% du PIB", 0, weight=0.0),
        ],
        experts=[
            ("ONU Tourisme — baromètre", "https://www.untourism.int/un-tourism-tourism-dashboard"),
            ("WEF — Travel & Tourism Development Index", "https://www.weforum.org/publications/travel-tourism-development-index-2024/"),
        ],
    ),
]

AGENT_BY_ID = {a["id"]: a for a in AGENTS}


def pc_scale(ind: dict) -> float | None:
    pc = ind.get("per_capita")
    if pc is True:
        return 1e6
    return float(pc) if pc else None

# Agents transverses (non notés directement dans le score actuel)
META_AGENTS = [
    dict(id="demographie", nom="Démographie & Prospective 2100", icone="🔭",
         description="Projections ONU à 2100 (population, vieillissement, espérance de vie), trajectoires ML des indicateurs, indice d'avenir."),
    dict(id="elections", nom="Pronostics électoraux & risque politique", icone="🗳️",
         description="Modèle ML de risque de choc politique à 3 ans, agrégateur de sondages, calendrier électoral, simulateur."),
    dict(id="veille", nom="Veille actualités & déclarations", icone="📰",
         description="Titres de presse récents, déclarations politiques, tonalité médiatique (Google News, GDELT)."),
    dict(id="synthese", nom="Synthèse & consensus", icone="🧠",
         description="Fusionne les agents, calcule les scores, rédige l'analyse pays et les rapports."),
]

# Indicateurs auxiliaires (non notés) nécessaires aux calculs et à la prospective
AUX_INDICATORS = [
    I("sp_pop_totl", "Population totale", "habitants", 0),
    I("sp_dyn_tfrt_in", "Fécondité", "enfants par femme", 0),
    I("sp_pop_65up_to_zs", "Population de 65 ans et +", "%", 0),
    I("sp_urb_totl_in_zs", "Population urbaine", "%", 0),
    I("sp_pop_grow", "Croissance démographique", "%/an", 0),
    I("ny_gdp_mktp_cd", "PIB (courant)", "$", 0),
    I("pa_nus_fcrf", "Taux de change officiel", "monnaie locale / $", 0),
    I("pop", "Population (ONU, projections jusqu'en 2100)", "habitants", 0, src="gm", origin="pop"),
    I("lex", "Espérance de vie (projections jusqu'en 2100)", "ans", 0, src="gm", origin="lex"),
    I("electoplx_eiu", "Processus électoral et pluralisme (EIU)", "0-100", 0, src="gm", origin="eiu"),
    I("v2cacamps", "Polarisation politique (V-Dem)", "échelle V-Dem", 0, src="vdem"),
    I("v2smpolsoc", "Polarisation de la société (V-Dem)", "échelle V-Dem", 0, src="vdem"),
    I("v2x_regime", "Régime politique (V-Dem, 0 autocratie fermée → 3 démocratie libérale)", "0-3", 0, src="vdem"),
    I("v2xel_frefair", "Élections libres et équitables", "indice 0-1", 0, src="vdem"),
    I("v2x_polyarchy", "Démocratie électorale (V-Dem)", "indice 0-1", 0, src="vdem"),
    I("v2xnp_regcorr", "Corruption du régime", "indice 0-1", 0, src="vdem"),
    I("v2x_execorr", "Corruption de l'exécutif", "indice 0-1", 0, src="vdem"),
]

# Poids de chaque composante de l'indice d'avenir (prospective 2100)
FUTURE_WEIGHTS = {
    "demographie": 0.20,   # dynamique et structure par âge 2050-2100
    "capital_humain": 0.20,  # éducation/PISA + trajectoire santé
    "institutions": 0.20,  # niveau et tendance 20 ans gouvernance/justice
    "climat_ressources": 0.20,  # renouvelables, eau, exposition, tendance émissions
    "dynamique_eco": 0.20,  # croissance tendancielle, innovation
}

GLOBAL_MIX = {"actuel": 0.75, "avenir": 0.25}

# Seuils de couleur de la carte (score global /100)
COLOR_THRESHOLDS = {"vert": 65, "orange": 45}


def all_indicators():
    """Liste dédupliquée (code, src) -> définition."""
    seen = {}
    for a in AGENTS:
        for ind in a["indicateurs"]:
            seen.setdefault((ind["src"], ind["code"]), ind)
    for ind in AUX_INDICATORS:
        seen.setdefault((ind["src"], ind["code"]), ind)
    return seen
