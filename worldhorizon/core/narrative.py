"""Agent Synthèse — rédaction de l'analyse pays (tous les aspects), à partir des données,
des tendances sur 50 ans, des comparaisons régionales, du ML et des références d'experts.

Le texte est généré de façon déterministe (aucune clé API nécessaire) ; core/llm.py peut
ensuite produire une version « avis d'expert » enrichie si un modèle de langage est configuré.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .catalog import AGENTS, AGENT_BY_ID, GM_ORIGIN, SOURCE_LABELS, SOURCE_URLS, all_indicators, pc_scale
from .countries import slugs
from .scoring import CURRENT_YEAR, level_of

COLOR_LABEL = {"vert": "situation solide (vert)", "orange": "situation sous surveillance (orange)",
               "rouge": "situation critique (rouge)", "gris": "données insuffisantes (gris)"}

# Lecture d'expert par pilier : (si score élevé, si score faible)
READINGS = {
    "sante": ("Le système de santé assure une longévité élevée et une faible mortalité évitable ; l'enjeu devient le coût du vieillissement et des maladies chroniques.",
              "Les résultats sanitaires restent en deçà du potentiel : mortalité infantile et maternelle, accès à l'eau et aux soins sont des priorités qui conditionnent la productivité future."),
    "education": ("Le capital humain est un atout majeur : les compétences mesurées (PISA, scolarisation) préparent une main-d'œuvre adaptable aux métiers de demain.",
                  "Les acquis scolaires sont insuffisants : la pauvreté des apprentissages et la faible scolarisation dans le secondaire et le supérieur limitent la montée en gamme de l'économie."),
    "economie": ("Les fondamentaux macroéconomiques sont solides (niveau de vie, emploi, stabilité des prix), ce qui donne des marges de manœuvre en cas de choc.",
                 "L'économie est vulnérable : niveau de vie faible ou instable, inflation, chômage ou endettement réduisent la capacité de l'État à investir et à protéger."),
    "finance": ("Le système financier est profond et capable de financer l'investissement ; les marchés de capitaux jouent un rôle significatif.",
                "Le financement de l'économie est contraint : faible crédit au secteur privé, marchés boursiers peu développés et coût du crédit élevé freinent l'investissement."),
    "business": ("L'environnement des affaires est favorable : règles prévisibles, infrastructures et attractivité pour les investisseurs.",
                 "Créer et développer une entreprise reste difficile : imprévisibilité réglementaire, coût du crédit, infrastructures (électricité, numérique) à renforcer."),
    "environnement": ("La trajectoire environnementale est relativement maîtrisée (émissions, renouvelables, ressources en eau).",
                      "Les pressions environnementales sont fortes : émissions élevées ou stress hydrique et exposition aux aléas climatiques constituent des risques de long terme."),
    "agriculture": ("La sécurité alimentaire est assurée, avec une agriculture productive.",
                    "La sécurité alimentaire est fragile : sous-alimentation, faibles rendements ou dépendance aux importations exposent la population aux chocs de prix."),
    "gouvernance": ("Les institutions démocratiques sont solides : élections libres, pluralisme et libertés garantissent l'alternance pacifique.",
                    "La qualité démocratique est faible : élections contestées ou non libres, libertés restreintes ; le pouvoir est peu soumis aux contre-pouvoirs."),
    "impunite": ("La justice est indépendante et ses décisions respectées : le sentiment d'impunité est faible et la confiance dans l'État de droit élevée.",
                 "Le sentiment d'impunité est élevé : la justice contrôle mal l'exécutif, ses décisions sont peu respectées et l'égalité devant la loi n'est pas garantie — un facteur clé de défiance et de vote sanction."),
    "corruption": ("La corruption est perçue comme limitée ; la transparence publique soutient la confiance et l'investissement.",
                   "La corruption est un problème systémique (politique, administrative, exécutif) : elle détourne des ressources, décourage l'investissement et alimente la colère sociale."),
    "securite": ("Le niveau de sécurité est élevé : faible violence criminelle, absence de conflit armé.",
                 "L'insécurité est un frein majeur : violence criminelle, conflits armés ou déplacements de population pèsent sur la vie quotidienne, l'économie et le tourisme."),
    "social": ("Le bien-être déclaré est élevé et les inégalités contenues : la cohésion sociale est un atout.",
               "Le bien-être social est dégradé : pauvreté, inégalités ou chômage des jeunes nourrissent frustration et migrations."),
    "innovation": ("Le pays est un acteur de l'innovation (R&D, brevets, recherche, numérique), ce qui soutient la productivité future.",
                   "La capacité d'innovation est limitée : faible effort de R&D et diffusion numérique incomplète réduisent les gains de productivité."),
    "tourisme": ("L'attractivité internationale est forte : le tourisme est une source importante de devises et de rayonnement.",
                 "Le potentiel touristique est sous-exploité ou freiné (sécurité, infrastructures, image)."),
}


# --------------------------------------------------------------------------- formats
def fnum(v, dec=None) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n.d."
    a = abs(v)
    if dec is None:
        dec = 0 if a >= 100 else (1 if a >= 10 else 2)
    s = f"{v:,.{dec}f}".replace(",", " ").replace(".", ",")
    return s


def fbig(v) -> str:
    if v is None or not math.isfinite(v):
        return "n.d."
    for t, w in ((1e12, "billions"), (1e9, "milliards"), (1e6, "millions")):
        if abs(v) >= t:
            return f"{fnum(v / t, 1)} {w}"
    return fnum(v, 0)


def ordinal(n: int) -> str:
    return "1er" if n == 1 else f"{n}e"


IND = all_indicators()


def ind_def(code: str) -> dict:
    for (src, c), d in IND.items():
        if c == code:
            return d
    return {"label": code, "unit": "", "src": "wdi", "direction": 0}


def source_of(d: dict) -> tuple[str, str]:
    label = SOURCE_LABELS.get(d["src"], d["src"])
    if d.get("origin"):
        label = f"{GM_ORIGIN.get(d['origin'], d['origin'])} (via {label})"
    url = SOURCE_URLS.get(d["src"], "").format(code=d["code"].upper().replace("_", "."))
    return label, url


def fval(code: str, v) -> str:
    d = ind_def(code)
    if d.get("per_capita"):
        return f"{fnum(v)} {d['unit']}"
    if "habitants" in d["unit"]:
        return fbig(v) + " hab."
    if d["unit"] == "$":
        return f"{fnum(v, 0)} $"
    return f"{fnum(v)} {d['unit']}".strip()


# --------------------------------------------------------------------------- piliers
def pillar_rank(st, pid: str, iso: str):
    s = st.pillars[pid].dropna()
    if iso not in s.index:
        return None, len(s)
    return int(s.rank(ascending=False, method="min")[iso]), len(s)


def pillar_history(st, iso: str, pid: str) -> pd.Series:
    h = st.pillars_hist[(st.pillars_hist.iso3 == iso) & (st.pillars_hist.pillar == pid)]
    return h.set_index("year")["score"].sort_index()


def _trend_sentence(h: pd.Series) -> str:
    if len(h) < 5:
        return ""
    last_y = int(h.index.max())
    h = h.dropna()
    if len(h) < 5:
        return ""
    last_y = int(h.index.max())
    parts = []
    for span in (10, 25):
        y = last_y - span
        past = h[h.index <= y]
        if len(past):
            d = h.iloc[-1] - past.iloc[-1]
            verb = "progressé" if d > 2 else ("reculé" if d < -2 else "peu varié")
            parts.append(f"sur {span} ans le score a {verb} ({'+' if d >= 0 else ''}{fnum(d, 1)} pts)")
    first_y = int(h.index.min())
    if last_y - first_y >= 30:
        parts.append(f"depuis {first_y} : {fnum(h.iloc[0], 0)} → {fnum(h.iloc[-1], 0)}/100")
    return ("Évolution : " + " ; ".join(parts) + ".") if parts else ""


def per_capita_series(st, iso: str, code: str) -> pd.Series:
    s = st.series(iso, code)
    k = pc_scale(ind_def(code))
    if k:
        pop = st.series(iso, "sp_pop_totl")
        s = (s / pop * k)
    return s.dropna()


def _long_run(st, iso: str, code: str) -> str:
    s = per_capita_series(st, iso, code)
    if len(s) < 2:
        return ""
    y0, y1 = int(s.index.min()), int(s.index.max())
    if y1 - y0 < 15:
        return ""
    return f"{ind_def(code)['label']} : {fval(code, s.iloc[0])} en {y0} → {fval(code, s.iloc[-1])} en {y1}."


def pillar_analysis(st, iso: str, pid: str) -> dict:
    a = AGENT_BY_ID[pid]
    c = st.countries.set_index("iso3").loc[iso]
    score = st.pillars.at[iso, pid] if iso in st.pillars.index else np.nan
    rank, n = pillar_rank(st, pid, iso)
    reg = st.countries[st.countries.region == c["region"]]["iso3"]
    reg_mean = st.pillars.loc[st.pillars.index.intersection(reg), pid].mean()
    world_mean = st.pillars[pid].mean()

    ind = st.indicators[(st.indicators.iso3 == iso) & (st.indicators.pillar == pid)].copy()
    rows = []
    for _, r in ind.iterrows():
        d = ind_def(r.indicator)
        if d.get("weight", 1) == 0:
            continue
        reg_vals = st.indicators[(st.indicators.indicator == r.indicator) & st.indicators.iso3.isin(reg)]["value"]
        world_vals = st.indicators[st.indicators.indicator == r.indicator]["value"]
        src, url = source_of(d)
        rows.append(dict(code=r.indicator, label=d["label"], unit=d["unit"], value=r.value, year=r.year,
                         score=r.score, reg_median=reg_vals.median(), world_median=world_vals.median(),
                         direction=d["direction"], source=src, url=url))
    tab = pd.DataFrame(rows)

    texte = []
    if np.isfinite(score):
        texte.append(f"Score {fnum(score, 0)}/100 — niveau {level_of(score)}"
                     + (f", {ordinal(rank)} sur {n}" if rank else "") + ".")
        if np.isfinite(reg_mean):
            ecart = score - reg_mean
            pos = "au-dessus" if ecart > 3 else ("en dessous" if ecart < -3 else "proche")
            texte.append(f"{pos.capitalize()} de la moyenne de la région {c['region_fr']} ({fnum(reg_mean, 0)}/100)"
                         f" et de la moyenne mondiale ({fnum(world_mean, 0)}/100)." if pos != "proche" else
                         f"Proche de la moyenne régionale ({fnum(reg_mean, 0)}/100) ; moyenne mondiale {fnum(world_mean, 0)}/100.")
    else:
        texte.append("Données insuffisantes pour noter ce pilier de façon fiable ; les indicateurs disponibles sont présentés ci-dessous.")

    forces, faiblesses = [], []
    if len(tab):
        t = tab.dropna(subset=["score"]).sort_values("score")
        for _, r in t.tail(2).iloc[::-1].iterrows():
            if r.score >= 55:
                forces.append(f"{r.label} : {fval(r.code, r.value)} ({int(r.year)}), médiane régionale {fval(r.code, r.reg_median)}")
        for _, r in t.head(2).iterrows():
            if r.score < 45:
                faiblesses.append(f"{r.label} : {fval(r.code, r.value)} ({int(r.year)}), médiane régionale {fval(r.code, r.reg_median)}")
    if forces:
        texte.append("Points forts — " + " ; ".join(forces) + ".")
    if faiblesses:
        texte.append("Points faibles — " + " ; ".join(faiblesses) + ".")

    h = pillar_history(st, iso, pid)
    tr = _trend_sentence(h)
    if tr:
        texte.append(tr)
    key = next((i["code"] for i in a["indicateurs"] if i["src"] != "pisa" and i["weight"] >= 1), None)
    lr = _long_run(st, iso, key) if key else ""
    if lr:
        texte.append("Repère historique — " + lr)
    p = st.projections[(st.projections.iso3 == iso) & (st.projections.horizon == 2050)]
    if len(p) and pid in p and np.isfinite(score):
        pv = p.set_index("scenario")[pid]
        if pd.notna(pv.get("central")):
            texte.append(f"Projection 2050 (tendance amortie) : {fnum(pv['central'], 0)}/100, fourchette "
                         f"{fnum(pv.get('pessimiste'), 0)}–{fnum(pv.get('optimiste'), 0)}.")
    if np.isfinite(score):
        hi, lo = READINGS[pid]
        texte.append("Lecture : " + (hi if score >= 60 else lo if score < 45 else
                                     "Situation intermédiaire : " + lo[0].lower() + lo[1:]))
    sl = slugs(c.to_dict() | {"iso3": iso})
    experts = []
    for nm, url in a["experts"]:
        try:
            experts.append((nm, url.format(**sl)))
        except (KeyError, IndexError):
            experts.append((nm, url))
    return dict(id=pid, nom=a["nom"], icone=a["icone"], score=score, rang=rank, n=n, niveau=level_of(score),
                reg_mean=reg_mean, world_mean=world_mean, texte=texte, forces=forces, faiblesses=faiblesses,
                indicateurs=tab, historique=h, experts=experts, description=a["description"])


# --------------------------------------------------------------------------- pays
def country_analysis(st, iso: str) -> dict:
    c = st.countries.set_index("iso3").loc[iso]
    m = st.master.loc[iso] if iso in st.master.index else pd.Series(dtype=float)
    name = c["name_fr"]
    pillars = {a["id"]: pillar_analysis(st, iso, a["id"]) for a in AGENTS}
    scored = {k: v for k, v in pillars.items() if np.isfinite(v["score"])}
    top = sorted(scored.values(), key=lambda p: -p["score"])[:3]
    bottom = sorted(scored.values(), key=lambda p: p["score"])[:3]
    sg = m.get("score_global", np.nan)
    n_cls = int(m.get("n_classes", 0) or 0)

    synth = []
    if pd.notna(sg):
        synth.append(f"{name} obtient un score global de {fnum(sg, 0)}/100 ({COLOR_LABEL[m['couleur']]}), "
                     f"au {ordinal(int(m['rang']))} rang sur {n_cls} pays et territoires évalués. Ce score combine la situation "
                     f"actuelle ({fnum(m['score_actuel'], 0)}/100, quatorze piliers) et l'indice d'avenir à l'horizon 2100 "
                     f"({fnum(m.get('indice_avenir'), 0)}/100). Fiabilité des données : {fnum(m.get('fiabilite'), 0)} %.")
    else:
        synth.append(f"{name} ne peut pas recevoir de score global fiable : trop peu de séries publiques internationales "
                     f"couvrent ce territoire (fiabilité {fnum(m.get('fiabilite'), 0)} %). Les indicateurs disponibles sont présentés ci-dessous.")
    if top:
        synth.append("Atouts principaux : " + ", ".join(f"{p['nom'].lower()} ({fnum(p['score'], 0)})" for p in top) + ".")
    if bottom:
        synth.append("Fragilités principales : " + ", ".join(f"{p['nom'].lower()} ({fnum(p['score'], 0)})" for p in bottom) + ".")
    gh = st.global_hist[st.global_hist.iso3 == iso].set_index("year")["score_actuel"].sort_index()
    if len(gh) > 10:
        y_last = int(gh.index.max())
        past = gh[gh.index <= y_last - 20]
        if len(past):
            d = gh.iloc[-1] - past.iloc[-1]
            synth.append(f"Sur vingt ans, la situation d'ensemble a {'progressé' if d > 2 else 'reculé' if d < -2 else 'peu évolué'} "
                         f"({fnum(past.iloc[-1], 0)} → {fnum(gh.iloc[-1], 0)}/100).")
    if pd.notna(m.get("risque_politique")):
        p3 = m.get("proba_recul_democratique_3ans")
        synth.append(f"Le risque politique est {str(m['niveau_risque']).lower()} (indice {fnum(m['risque_politique'], 0)}/100"
                     + (f" ; probabilité estimée par le modèle d'un recul démocratique marqué d'ici 3 ans : {fnum(p3 * 100, 0)} %" if pd.notna(p3) else "")
                     + ").")
    f = st.future.loc[iso] if iso in st.future.index else None
    if f is not None and pd.notna(f.get("pop_2100")) and pd.notna(f.get("pop_2025")):
        synth.append(f"Selon les projections des Nations unies, la population passerait de {fbig(f['pop_2025'])} (2025) à "
                     f"{fbig(f['pop_2050'])} en 2050 et {fbig(f['pop_2100'])} en 2100 ({'+' if f['var_pop_2100_pct'] >= 0 else ''}{fnum(f['var_pop_2100_pct'], 0)} %).")
    if iso in st.peers.index:
        pe = [st.name(x) for x in str(st.peers.at[iso, "pays_proches"]).split(";")]
        synth.append("Pays au profil le plus proche (apprentissage non supervisé sur les 14 piliers) : " + ", ".join(pe) + ".")

    return dict(iso3=iso, nom=name, pays=c, master=m, synthese=synth, piliers=pillars, swot=swot(st, iso, pillars),
                avenir=future_text(st, iso), politique=politics_text(st, iso), sources=sources_table(pillars),
                historique_global=gh)


def swot(st, iso: str, pillars: dict) -> dict:
    S, W, O, T = [], [], [], []
    for p in pillars.values():
        if not np.isfinite(p["score"]):
            continue
        if p["score"] >= 65:
            S.append(f"{p['nom']} ({fnum(p['score'], 0)}/100)")
        elif p["score"] < 40:
            W.append(f"{p['nom']} ({fnum(p['score'], 0)}/100)")
        h = p["historique"]
        if len(h) > 12:
            d = h.iloc[-1] - h[h.index <= h.index.max() - 10].iloc[-1] if len(h[h.index <= h.index.max() - 10]) else 0
            if d >= 5:
                O.append(f"Dynamique positive : {p['nom'].lower()} (+{fnum(d, 0)} pts en 10 ans)")
            elif d <= -5:
                T.append(f"Dégradation : {p['nom'].lower()} ({fnum(d, 0)} pts en 10 ans)")
    f = st.future.loc[iso] if iso in st.future.index else None
    if f is not None:
        v = f.get("var_pop_2100_pct")
        if pd.notna(v):
            if v > 80:
                O.append(f"Dividende démographique potentiel (population +{fnum(v, 0)} % d'ici 2100) — à condition d'investir dans l'éducation et l'emploi")
                T.append("Pression démographique sur l'emploi, l'eau, l'alimentation et les villes")
            elif v < -25:
                T.append(f"Déclin démographique marqué ({fnum(v, 0)} % d'ici 2100) : vieillissement, financement des retraites, pénurie de main-d'œuvre")
            else:
                O.append("Démographie relativement équilibrée à l'horizon 2100")
        if pd.notna(f.get("croissance_tendancielle_pct")) and f["croissance_tendancielle_pct"] > 3:
            O.append(f"Croissance tendancielle soutenue du revenu par habitant ({fnum(f['croissance_tendancielle_pct'], 1)} %/an sur 20 ans)")
    m = st.master.loc[iso] if iso in st.master.index else None
    if m is not None and str(m.get("niveau_risque")) == "Élevé":
        T.append("Risque politique élevé (instabilité, recul démocratique possible)")
    env = pillars["environnement"]["indicateurs"]
    if len(env):
        ws = env[env.code == "er_h2o_fwst_zs"]
        if len(ws) and ws.iloc[0].value > 70:
            T.append(f"Stress hydrique sévère ({fnum(ws.iloc[0].value, 0)} % des ressources prélevées)")
    S = sorted(S, key=lambda x: -float(x.split("(")[-1].split("/")[0]))[:6]
    W = sorted(W, key=lambda x: float(x.split("(")[-1].split("/")[0]))[:6]
    return dict(Forces=S or ["—"], Faiblesses=W or ["—"], Opportunités=O or ["—"], Menaces=T or ["—"])


COMP_LABEL = {"demographie": "Démographie", "capital_humain": "Capital humain", "institutions": "Institutions",
              "climat_ressources": "Climat & ressources", "dynamique_eco": "Dynamique économique"}


def future_text(st, iso: str) -> dict:
    if iso not in st.future.index:
        return dict(texte=["Prospective indisponible pour ce territoire."], composantes={})
    f = st.future.loc[iso]
    comps = {COMP_LABEL[k]: f.get(k) for k in COMP_LABEL}
    t = []
    if pd.notna(f.get("indice_avenir")):
        t.append(f"Indice d'avenir (horizon 2100) : {fnum(f['indice_avenir'], 0)}/100. Il ne s'agit pas d'une prédiction "
                 "mais d'une mesure des fondations structurelles qui déterminent la trajectoire de long terme : "
                 + ", ".join(f"{k.lower()} {fnum(v, 0)}" for k, v in comps.items() if pd.notna(v)) + ".")
    if pd.notna(f.get("score_2050_central")):
        t.append(f"Si les tendances des 25 dernières années se prolongent en s'atténuant, le score de situation atteindrait "
                 f"{fnum(f['score_2050_central'], 0)}/100 en 2050 (scénario pessimiste {fnum(f['score_2050_pessimiste'], 0)}, "
                 f"optimiste {fnum(f['score_2050_optimiste'], 0)}).")
    best = max(((k, v) for k, v in comps.items() if pd.notna(v)), key=lambda kv: kv[1], default=None)
    worst = min(((k, v) for k, v in comps.items() if pd.notna(v)), key=lambda kv: kv[1], default=None)
    if best and worst:
        t.append(f"Le principal levier d'avenir est : {best[0].lower()} ; le principal frein : {worst[0].lower()}. "
                 "Un avenir brillant sur le siècle suppose d'agir en priorité sur ce frein.")
    if pd.notna(f.get("fecondite")):
        t.append(f"Fécondité actuelle : {fnum(f['fecondite'], 1)} enfant(s) par femme ; espérance de vie projetée en 2100 : "
                 f"{fnum(f.get('lex_2100'), 0)} ans (ONU).")
    t.append("Repères pour 2100 : projections de population ONU (World Population Prospects), scénarios climatiques du GIEC "
             "(Atlas interactif), trajectoires des indicateurs WorldHorizon recalculées à chaque mise à jour.")
    return dict(texte=t, composantes=comps)


def politics_text(st, iso: str) -> dict:
    if iso not in st.risk.index:
        return dict(texte=["Indice de risque politique indisponible."], facteurs=[])
    r = st.risk.loc[iso]
    t = [f"Indice de risque politique : {fnum(r['indice_risque_politique'], 0)}/100 ({r['niveau']})."]
    if pd.notna(r.get("proba_recul_democratique_3ans")):
        t.append(f"Probabilité d'un recul démocratique marqué dans les 3 ans (modèle de gradient boosting entraîné sur "
                 f"plus de 40 ans de données V-Dem et Banque mondiale) : {fnum(r['proba_recul_democratique_3ans'] * 100, 0)} %.")
    t.append(f"Fragilité structurelle (sécurité, impunité, corruption, économie, social) : {fnum(r['fragilite_structurelle'], 0)}/100.")
    poly = st.series(iso, "v2x_polyarchy").dropna()
    if len(poly) and poly.iloc[-1] < 0.4:
        t.append("Note de lecture : le modèle estime un risque de *recul* démocratique ; dans un régime déjà peu démocratique "
                 f"(démocratie électorale V-Dem {fnum(poly.iloc[-1], 2)}), cette probabilité est mécaniquement faible et c'est "
                 "la fragilité structurelle qui mesure le mieux le risque d'instabilité.")
    pol = st.series(iso, "v2cacamps").dropna()
    if len(pol):
        t.append(f"Polarisation politique (V-Dem, {int(pol.index.max())}) : {fnum(pol.iloc[-1], 2)} "
                 f"(plus la valeur est élevée, plus les camps politiques sont antagonistes ; centile mondial {fnum(r.get('polarisation_centile'), 0)}).")
    imp = st.pillars.at[iso, "impunite"] if iso in st.pillars.index else np.nan
    infl = st.indicators[(st.indicators.iso3 == iso) & (st.indicators.indicator == "fp_cpi_totl_zg")]
    hap = st.indicators[(st.indicators.iso3 == iso) & (st.indicators.indicator == "hapiscore_whr")]
    drivers = []
    if np.isfinite(imp):
        drivers.append(f"sentiment d'impunité {'faible' if imp >= 60 else 'élevé' if imp < 45 else 'modéré'} (pilier justice {fnum(imp, 0)}/100)")
    if len(infl):
        drivers.append(f"inflation {fnum(infl.iloc[0].value, 1)} % ({int(infl.iloc[0].year)})")
    if len(hap):
        drivers.append(f"bonheur déclaré {fnum(hap.iloc[0].value, 0)}/100")
    if drivers:
        t.append("Déterminants du vote à surveiller : " + ", ".join(drivers) + ". Une impunité perçue élevée combinée à une "
                 "inflation forte est historiquement associée aux votes sanction et à la montée des partis antisystème.")
    fac = []
    if isinstance(r.get("facteurs"), str):
        fac = r["facteurs"].split(";")
    else:
        from .paths import ML_DIR
        p = ML_DIR / "political_risk.csv"
        if p.exists():
            pr = pd.read_csv(p).set_index("iso3")
            if iso in pr.index and isinstance(pr.at[iso, "facteurs"], str):
                fac = pr.at[iso, "facteurs"].split(";")
    from .ml_engine import FEATURE_LABELS
    facteurs = []
    for x in fac:
        if ":" in x:
            k, v = x.split(":")
            facteurs.append((FEATURE_LABELS.get(k, k), float(v)))
    return dict(texte=t, facteurs=facteurs)


def sources_table(pillars: dict) -> pd.DataFrame:
    rows = []
    for p in pillars.values():
        t = p["indicateurs"]
        for _, r in t.iterrows():
            rows.append(dict(pilier=p["nom"], indicateur=r.label, code=r.code, annee=int(r.year), source=r.source, lien=r.url))
    rows.append(dict(pilier="Prospective", indicateur="Population et espérance de vie projetées 2025-2100", code="pop, lex",
                     annee=2100, source="Nations unies — World Population Prospects (via Gapminder)", lien="https://population.un.org/wpp/"))
    return pd.DataFrame(rows).drop_duplicates(["code", "pilier"])


def fact_sheet(an: dict) -> str:
    """Résumé chiffré compact pour le modèle de langage (aucune donnée inventée)."""
    lines = [f"PAYS : {an['nom']} ({an['iso3']}) — région {an['pays']['region_fr']}, {an['pays']['income_fr']}."]
    lines += an["synthese"]
    for p in an["piliers"].values():
        lines.append(f"\n## {p['nom']} — score {fnum(p['score'], 0)}/100")
        for _, r in p["indicateurs"].iterrows():
            lines.append(f"- {r.label} : {fval(r.code, r.value)} ({int(r.year)}) ; médiane régionale {fval(r.code, r.reg_median)} ; source {r.source}")
        lines += [x for x in p["texte"] if x.startswith(("Évolution", "Repère", "Projection"))]
    lines.append("\n## Prospective"); lines += an["avenir"]["texte"]
    lines.append("\n## Politique"); lines += an["politique"]["texte"]
    return "\n".join(lines)
