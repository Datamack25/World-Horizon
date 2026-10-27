"""WorldHorizon — plateforme d'analyse holistique des pays et territoires.
Lancer : streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from core import images as im
from core.catalog import AGENT_BY_ID, AGENTS, COLOR_THRESHOLDS, GLOBAL_MIX, META_AGENTS
from core.countries import search
from core.elections import (CAL, POLLS, aggregate_polls, load_calendar, load_polls, upcoming, what_if)
from core.narrative import COLOR_LABEL, country_analysis, fnum, fval
from core.paths import AGENTS_DIR, ASSETS, ML_DIR, ROOT, SCORES

st.set_page_config(page_title="WorldHorizon — analyse des pays", page_icon="🌍", layout="wide")
st.markdown(f"<style>{(ASSETS / 'style.css').read_text()}</style>", unsafe_allow_html=True)

HEX = {"vert": "#2E8B57", "orange": "#E0952B", "rouge": "#C8463D", "gris": "#B7BEC6"}
LABEL_COUL = {"vert": "Vert — situation solide", "orange": "Orange — sous surveillance",
              "rouge": "Rouge — situation critique", "gris": "Gris — données insuffisantes"}


# --------------------------------------------------------------------------- données
@st.cache_resource(show_spinner="Chargement des données des agents…")
def get_store():
    from core.store import load_store
    return load_store()


@st.cache_resource(show_spinner="Rédaction de l'analyse…", max_entries=64)
def get_analysis(iso: str, _ver: str):
    return country_analysis(get_store(), iso)


def data_version() -> str:
    p = SCORES / "country_scores_master.csv"
    return str(p.stat().st_mtime) if p.exists() else "0"


def flag(iso2) -> str:
    if not isinstance(iso2, str) or len(iso2) != 2 or not iso2.isalpha():
        return "🏳️"
    return "".join(chr(0x1F1E6 + ord(c) - 65) for c in iso2.upper())


def color_class(score) -> str:
    if score is None or not np.isfinite(score):
        return "gris"
    return "vert" if score >= COLOR_THRESHOLDS["vert"] else "orange" if score >= COLOR_THRESHOLDS["orange"] else "rouge"


def kpi_html(items) -> str:
    cells = "".join(f'<div class="wh-kpi {c}"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l, c in items)
    return f'<div class="wh-kpis">{cells}</div>'


store = get_store()
if store.master.empty:
    st.error("Aucune donnée calculée. Lancez `python -m core.pipeline` ou le bouton « Mettre à jour les informations ».")
M = store.master
OPTIONS = store.countries.sort_values("name_fr")
ISO_LIST = list(OPTIONS["iso3"])
LABELS = {r.iso3: f"{flag(r.iso2)}  {r.name_fr}" for r in OPTIONS.itertuples()}


def go_country(iso: str):
    st.session_state["iso"] = iso
    st.switch_page(PAGES["pays"])


# --------------------------------------------------------------------------- barre latérale
def sidebar():
    with st.sidebar:
        st.markdown("<h2 style='color:#fff;margin-bottom:0'>WorldHorizon</h2>"
                    "<p style='color:#A9BBCB;margin-top:0'>Situation et avenir de "
                    f"{len(store.countries)} pays et territoires</p>", unsafe_allow_html=True)
        cur = st.session_state.get("iso")
        sel = st.selectbox("Rechercher un pays ou un territoire", ISO_LIST, index=ISO_LIST.index(cur) if cur in ISO_LIST else None,
                           format_func=lambda i: LABELS.get(i, i), placeholder="Tapez un nom : Haïti, Japon, Sénégal…",
                           key="sb_search")
        if sel and sel != cur:
            go_country(sel)
        st.divider()
        log = pd.read_csv(AGENTS_DIR / "_runs_log.csv") if (AGENTS_DIR / "_runs_log.csv").exists() else pd.DataFrame()
        if len(log):
            last = pd.to_datetime(log["run_at"]).max()
            st.caption(f"Dernière collecte : {last:%d/%m/%Y %H:%M} UTC")
        update_panel(compact=True)


def update_panel(compact=False):
    from core import github_sync
    key = "c" if compact else "f"
    if not compact:
        c1, c2, c3 = st.columns(3)
        markets = c1.checkbox("Marchés financiers (Yahoo Finance)", True, key=f"m{key}")
        cal = c2.checkbox("Calendrier électoral (Wikipédia)", True, key=f"k{key}")
        push = c3.checkbox("Sauvegarder dans GitHub", github_sync.configured(), disabled=not github_sync.configured(), key=f"g{key}")
    else:
        markets, cal, push = False, False, github_sync.configured()
    if st.button("🔄 Mettre à jour les informations", type="primary", width="stretch", key=f"upd{key}"):
        import os
        from core.pipeline import run
        # V-Dem (≈1 Go en mémoire) est mis à jour par la GitHub Action hebdomadaire ; dans l'appli on garde l'historique.
        from core.llm import _secret
        if _secret("WH_LOAD_VDEM") != "1":
            os.environ["WH_SKIP_VDEM"] = "1"
        with st.status("Les agents collectent les données…", expanded=not compact) as s:
            bar = st.progress(0.0)
            try:
                res = run(fetch=True, markets=markets, calendar=cal, log=lambda m: st.write(m),
                          progress=lambda f, m: bar.progress(f, text=m))
                if push:
                    url = github_sync.commit_data()
                    st.write(f"💾 Données sauvegardées dans GitHub : {url}")
                s.update(label="Mise à jour terminée", state="complete")
            except Exception as e:
                s.update(label=f"Mise à jour interrompue : {e}", state="error")
                return
        st.cache_resource.clear()
        st.rerun()


# --------------------------------------------------------------------------- carte
@st.cache_data(show_spinner=False)
def geo_ids() -> set:
    p = ROOT / "data/ref/world.geo.json"
    return {f["id"] for f in json.loads(p.read_text())["features"]} if p.exists() else set()


def world_map(metric: str, year: int | None = None, height=560):
    df = M.reset_index().copy()
    if metric == "historique":
        gh = store.global_hist[store.global_hist.year == year].set_index("iso3")["score_actuel"]
        df["val"] = df["iso3"].map(gh)
        df["couleur"] = df["val"].map(color_class)
        title_val = f"Score de situation {year}"
    elif metric == "score_global":
        df["val"] = df["score_global"]; title_val = "Score global"
    else:
        df["val"] = df[metric]; title_val = METRICS[metric]
    df["nom"] = df["name_fr"]
    df["txt"] = df["val"].map(lambda v: fnum(v, 0) if pd.notna(v) else "n.d.")
    hover = "<b>%{customdata[0]}</b><br>" + title_val + " : %{customdata[1]}<br>Risque politique : %{customdata[2]}<extra></extra>"
    custom = np.stack([df["nom"], df["txt"], df["niveau_risque"].fillna("n.d.")], axis=1)
    discrete = metric in ("score_global", "historique")
    fig = go.Figure()
    if discrete:
        cats = df["couleur"].fillna("gris")
        for k in ("vert", "orange", "rouge", "gris"):
            sub = cats == k
            fig.add_trace(go.Choropleth(locations=df.loc[sub, "iso3"], z=[1] * int(sub.sum()), locationmode="ISO-3",
                                        colorscale=[[0, HEX[k]], [1, HEX[k]]], showscale=False, name=LABEL_COUL[k],
                                        customdata=custom[sub.values], hovertemplate=hover,
                                        marker_line_color="#ffffff", marker_line_width=0.4))
    else:
        rev = metric == "risque_politique"
        fig.add_trace(go.Choropleth(locations=df["iso3"], z=df["val"], locationmode="ISO-3", zmin=0, zmax=100,
                                    colorscale=[[0, HEX["vert"] if rev else HEX["rouge"]], [0.5, HEX["orange"]],
                                                [1, HEX["rouge"] if rev else HEX["vert"]]],
                                    customdata=custom, hovertemplate=hover, marker_line_color="#fff", marker_line_width=0.4,
                                    colorbar=dict(title="", thickness=10, len=0.6)))
    # petits territoires et îles : marqueurs ronds
    small = df[~df["iso3"].isin(geo_ids()) & df["latitude"].notna()] if "latitude" in df else df.head(0)
    if "latitude" not in df:
        small = df.merge(store.countries[["iso3", "latitude", "longitude"]], on="iso3")
        small = small[~small["iso3"].isin(geo_ids()) & small["latitude"].notna()]
    if len(small):
        cols = small["couleur"].map(HEX) if discrete else small["val"].map(
            lambda v: HEX[color_class(100 - v if metric == "risque_politique" else v)] if pd.notna(v) else HEX["gris"])
        fig.add_trace(go.Scattergeo(lat=small["latitude"], lon=small["longitude"], mode="markers",
                                    marker=dict(size=7, color=cols, line=dict(color="#fff", width=1)),
                                    customdata=np.stack([small["nom"], small["txt"], small["niveau_risque"].fillna("n.d.")], axis=1),
                                    hovertemplate=hover, showlegend=False, text=small["iso3"]))
    fig.update_geos(projection_type="natural earth", showframe=False, showcoastlines=False, showland=True,
                    landcolor="#DCE2DE", showocean=True, oceancolor="#F3F5F2", showcountries=False, lataxis_range=[-58, 85])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=False, dragmode=False)
    return fig


def map_click_iso(event) -> str | None:
    try:
        pts = event.selection.points
    except Exception:
        return None
    for p in pts:
        loc = p.get("location") or p.get("text")
        if loc:
            return loc
    return None


METRICS = {"score_global": "Score global (couleurs)", "score_actuel": "Situation actuelle",
           "indice_avenir": "Indice d'avenir 2100", "risque_politique": "Risque politique",
           **{f"p_{a['id']}": a["nom"] for a in AGENTS}, "historique": "Historique 1975 → aujourd'hui"}


# --------------------------------------------------------------------------- pages
def page_home():
    st.markdown("<div class='wh-hero'><h1 class='wh-title'>Où en est le monde, pays par pays ?</h1>"
                f"<p>{len(AGENTS)} agents analysent la santé, l'éducation, l'économie, la justice, la sécurité, le climat… "
                "sur 50 ans de données publiques, puis estiment l'avenir de chaque pays jusqu'en 2100.</p></div>",
                unsafe_allow_html=True)
    q = st.text_input("Rechercher un pays", placeholder="Rechercher un pays ou un territoire — ex. Haïti, Côte d'Ivoire, Corée du Sud",
                      label_visibility="collapsed")
    if q:
        res = search(q, store.countries, 8)
        if res.empty:
            st.info("Aucun pays ne correspond. Essayez le nom français ou anglais, ou le code ISO (ex. HTI).")
        else:
            cols = st.columns(min(len(res), 4))
            for i, r in enumerate(res.itertuples()):
                if cols[i % len(cols)].button(f"{flag(r.iso2)} {r.name_fr}", key=f"q{r.iso3}", width="stretch"):
                    go_country(r.iso3)
    n = M["couleur"].value_counts()
    counts = {k: int(n.get(k, 0)) for k in ("vert", "orange", "rouge", "gris")}
    components.html(f"""
    <div id="k" style="display:flex;gap:28px;flex-wrap:wrap;font-family:'Source Sans 3',sans-serif;color:#172A3A">
      {''.join(f'<div><div class="n" data-v="{v}" style="font:800 2.1rem Archivo,sans-serif;color:{HEX[k] if k != "gris" else "#6B7682"}">0</div>'
               f'<div style="color:#4A5D6E;font-size:.9rem">{LABEL_COUL[k].split(" — ")[1]}</div></div>' for k, v in counts.items())}
    </div>
    <script>
      const els=[...document.querySelectorAll('.n')];
      const still=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      els.forEach(e=>{{const t=+e.dataset.v; if(still){{e.textContent=t;return}}
        let s=null; const step=ts=>{{if(!s)s=ts; const p=Math.min((ts-s)/900,1);
        e.textContent=Math.round(t*(1-Math.pow(1-p,3))); if(p<1)requestAnimationFrame(step)}}; requestAnimationFrame(step);}});
    </script>""", height=78)
    c1, c2 = st.columns([3, 1])
    metric = c2.selectbox("Afficher sur la carte", list(METRICS), format_func=METRICS.get)
    year = None
    if metric == "historique":
        years = sorted(store.global_hist["year"].unique())
        year = c2.slider("Année", int(years[0]), int(years[-1]), int(years[-1]))
    c1.markdown("<div class='wh-legend'>" + "".join(f"<span style='--c:{HEX[k]}'>{LABEL_COUL[k]}</span>" for k in HEX)
                + f"</div><div class='wh-note'>Vert ≥ {COLOR_THRESHOLDS['vert']}, orange ≥ {COLOR_THRESHOLDS['orange']}, "
                "rouge en dessous. Cliquez sur un pays pour ouvrir son analyse.</div>", unsafe_allow_html=True)
    ev = st.plotly_chart(world_map(metric, year), width="stretch", on_select="rerun", selection_mode="points",
                         key=f"map_{metric}_{year}", config={"displayModeBar": False, "scrollZoom": False})
    iso = map_click_iso(ev)
    if iso and iso in ISO_LIST:
        go_country(iso)
    a, b = st.columns(2)
    top = M.dropna(subset=["score_global"]).sort_values("score_global", ascending=False)
    with a:
        st.subheader("En tête")
        st.dataframe(top.head(10)[["name_fr", "score_global", "score_actuel", "indice_avenir"]].round(0),
                     column_config={"name_fr": "Pays", "score_global": "Global", "score_actuel": "Actuel", "indice_avenir": "Avenir"},
                     hide_index=True, width="stretch")
    with b:
        st.subheader("En situation critique")
        st.dataframe(top.tail(10).iloc[::-1][["name_fr", "score_global", "score_actuel", "indice_avenir"]].round(0),
                     column_config={"name_fr": "Pays", "score_global": "Global", "score_actuel": "Actuel", "indice_avenir": "Avenir"},
                     hide_index=True, width="stretch")
    with st.expander("Les agents et ce qu'ils analysent"):
        rows = [(a["icone"] + " " + a["nom"], f"{a['poids']} %", a["description"]) for a in AGENTS]
        rows += [(a["icone"] + " " + a["nom"], "—", a["description"]) for a in META_AGENTS]
        st.dataframe(pd.DataFrame(rows, columns=["Agent", "Poids", "Ce qu'il mesure"]), hide_index=True, width="stretch")


def _radar_fig(series: dict[str, dict], height=420):
    fig = go.Figure()
    theta = [im.SHORT[a["id"]] for a in AGENTS]
    pal = ["#1F5C7A", "#C8463D", "#2E8B57", "#E0952B", "#6B4C9A", "#4A5D6E"]
    for i, (name, vals) in enumerate(series.items()):
        r = [vals.get(a["id"], None) for a in AGENTS]
        fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=theta + theta[:1], name=name, fill="toself" if i == 0 else None,
                                      line=dict(color=pal[i % len(pal)], width=2.2 if i == 0 else 1.6,
                                                dash=None if i < 6 and name != "Moyenne mondiale" else "dash"), opacity=0.9))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], tickfont_size=9), bgcolor="rgba(0,0,0,0)"), height=height,
                      margin=dict(l=40, r=40, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
    return fig


def page_country():
    iso = st.session_state.get("iso")
    if not iso:
        st.markdown("<div class='wh-hero'><h1 class='wh-title'>Analyse d'un pays</h1><p>Choisissez un pays ou un territoire dans la barre de recherche.</p></div>", unsafe_allow_html=True)
        sel = st.selectbox("Pays", ISO_LIST, index=None, format_func=LABELS.get, placeholder="Rechercher…")
        if sel:
            st.session_state["iso"] = sel; st.rerun()
        return
    an = get_analysis(iso, data_version())
    c = an["pays"]; m = an["master"]
    st.markdown(f"<div class='wh-hero'><h1 class='wh-title'>{flag(c['iso2'])} {an['nom']}</h1>"
                f"<p>{c['continent_fr']} · {c['region_fr']} · {c['income_fr']}"
                + (" · territoire" if str(c.get("territoire")) == "True" else "") + "</p></div>", unsafe_allow_html=True)
    rang = f"{int(m['rang'])}<span style='font-size:1rem'>e/{int(m['n_classes'])}</span>" if pd.notna(m.get("rang")) else "n.d."
    st.markdown(kpi_html([
        (fnum(m.get("score_global"), 0), "Score global /100", color_class(m.get("score_global"))),
        (fnum(m.get("score_actuel"), 0), "Situation actuelle", color_class(m.get("score_actuel"))),
        (fnum(m.get("indice_avenir"), 0), "Indice d'avenir 2100", color_class(m.get("indice_avenir"))),
        (rang, "Rang mondial", ""),
        (str(m.get("niveau_risque") or "n.d."), "Risque politique",
         {"Faible": "vert", "Modéré": "orange", "Élevé": "rouge"}.get(str(m.get("niveau_risque")), "gris")),
        (fnum(m.get("fiabilite"), 0) + " %", "Fiabilité des données", ""),
    ]), unsafe_allow_html=True)

    tabs = st.tabs(["Synthèse", "Tous les aspects", "Avenir 2050–2100", "Politique & élections", "Marchés",
                    "Actualités", "Avis d'expert", "Rapports PDF & images", "Sources", "🕵️ Intelligence économique"])
    with tabs[0]:
        a, b = st.columns([1.15, 1])
        with a:
            st.markdown("<div class='wh-prose'>" + "".join(f"<p>{t}</p>" for t in an["synthese"]) + "</div>", unsafe_allow_html=True)
            sw = an["swot"]
            colmap = {"Forces": HEX["vert"], "Faiblesses": HEX["rouge"], "Opportunités": "#1F5C7A", "Menaces": HEX["orange"]}
            st.markdown("<div class='wh-swot'>" + "".join(
                f"<div style='--c:{colmap[k]}'><h4>{k}</h4><ul>{''.join(f'<li>{x}</li>' for x in v)}</ul></div>" for k, v in sw.items())
                + "</div>", unsafe_allow_html=True)
        with b:
            pil = store.pillars.loc[iso].to_dict() if iso in store.pillars.index else {}
            st.plotly_chart(_radar_fig({an["nom"]: pil, "Moyenne mondiale": store.pillars.mean().to_dict()}), width="stretch")
            gh = an["historique_global"]
            if len(gh):
                f = px.line(gh, labels={"value": "score", "year": ""}, height=230)
                f.update_traces(line_color="#172A3A")
                f.add_hrect(y0=COLOR_THRESHOLDS["vert"], y1=100, fillcolor=HEX["vert"], opacity=0.07, line_width=0)
                f.add_hrect(y0=0, y1=COLOR_THRESHOLDS["orange"], fillcolor=HEX["rouge"], opacity=0.07, line_width=0)
                f.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0), yaxis_range=[0, 100],
                                title=dict(text="Score de situation depuis 1975", font_size=13), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(f, width="stretch")

    with tabs[1]:
        st.caption("Chaque domaine est analysé par un agent indépendant : constat chiffré, comparaison régionale, tendance sur 50 ans, projection et lecture d'expert.")
        for a in AGENTS:
            p = an["piliers"][a["id"]]
            cls = color_class(p["score"])
            with st.expander(f"{a['icone']} {p['nom']} — {fnum(p['score'], 0)}/100 · {p['niveau']}", expanded=False):
                l, r = st.columns([1.2, 1])
                with l:
                    st.markdown(f"<span class='wh-pill {cls}'>{fnum(p['score'], 0)}/100</span> "
                                f"<span class='wh-note'>{p['description']}</span>", unsafe_allow_html=True)
                    st.markdown("<div class='wh-prose'>" + "".join(f"<p>{t}</p>" for t in p["texte"]) + "</div>", unsafe_allow_html=True)
                    st.markdown("**Pour aller plus loin :** " + " · ".join(f"[{n}]({u})" for n, u in p["experts"]))
                with r:
                    t = p["indicateurs"]
                    if len(t):
                        show = pd.DataFrame({"Indicateur": t.label, "Valeur": [fval(cd, v) for cd, v in zip(t.code, t.value)],
                                             "Année": t.year.astype(int), "Médiane région": [fval(cd, v) for cd, v in zip(t.code, t.reg_median)],
                                             "Score": t.score.round(0)})
                        st.dataframe(show, hide_index=True, width="stretch")
                        lab2code = dict(zip(t.label, t.code))
                        code = lab2code[st.selectbox("Série longue", list(lab2code), key=f"s{a['id']}")]
                        from core.narrative import per_capita_series
                        s = per_capita_series(store, iso, code)
                        if len(s):
                            f = px.line(s, height=220, labels={"value": "", "year": ""})
                            f.update_traces(line_color=HEX.get(cls, "#1F5C7A"))
                            f.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
                            st.plotly_chart(f, width="stretch", key=f"c{a['id']}")
                    else:
                        st.info("Aucune donnée publique internationale récente pour ce domaine.")

    with tabs[2]:
        fut = an["avenir"]
        st.markdown("<div class='wh-prose'>" + "".join(f"<p>{t}</p>" for t in fut["texte"]) + "</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        proj = store.projections[store.projections.iso3 == iso]
        gh = an["historique_global"]
        with a:
            fig = go.Figure()
            if len(gh):
                fig.add_trace(go.Scatter(x=gh.index, y=gh.values, name="historique", line=dict(color="#172A3A", width=2.4)))
                for scen, col in (("pessimiste", HEX["rouge"]), ("central", "#1F5C7A"), ("optimiste", HEX["vert"])):
                    pp = proj[proj.scenario == scen].sort_values("horizon")
                    fig.add_trace(go.Scatter(x=[gh.index[-1]] + list(pp.horizon), y=[gh.values[-1]] + list(pp.score_actuel),
                                             name=f"scénario {scen}", line=dict(color=col, dash=None if scen == "central" else "dash")))
            fig.update_layout(height=330, yaxis_range=[0, 100], title="Score de situation : passé et scénarios 2035–2050",
                              margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")
        with b:
            pop = store.series(iso, "pop")
            if len(pop):
                f = go.Figure()
                f.add_trace(go.Scatter(x=pop.index[pop.index <= 2025], y=pop[pop.index <= 2025] / 1e6, name="observé", line=dict(color="#172A3A")))
                f.add_trace(go.Scatter(x=pop.index[pop.index >= 2025], y=pop[pop.index >= 2025] / 1e6, name="projection ONU", line=dict(color="#1F5C7A", dash="dash")))
                f.update_layout(height=330, title="Population (millions), 1970 → 2100", margin=dict(l=0, r=0, t=40, b=0),
                                paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h"))
                st.plotly_chart(f, width="stretch")
        if fut["composantes"]:
            comp = pd.Series(fut["composantes"]).dropna()
            f = px.bar(comp, orientation="h", range_x=[0, 100], height=240, labels={"value": "", "index": ""},
                       color=comp.map(lambda v: HEX[color_class(v)]), color_discrete_map="identity")
            f.update_layout(showlegend=False, title="Composantes de l'indice d'avenir", margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(f, width="stretch")

    with tabs[3]:
        pol = an["politique"]
        st.markdown("<div class='wh-prose'>" + "".join(f"<p>{t}</p>" for t in pol["texte"]) + "</div>", unsafe_allow_html=True)
        if pol["facteurs"]:
            fdf = pd.DataFrame(pol["facteurs"], columns=["Facteur", "Effet"])
            fdf["Effet"] *= 100
            f = px.bar(fdf, x="Effet", y="Facteur", orientation="h", height=260, labels={"Effet": "effet sur la probabilité (points)"},
                       color=fdf["Effet"].map(lambda v: HEX["rouge"] if v > 0 else HEX["vert"]), color_discrete_map="identity")
            f.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0), title="Ce qui pèse dans la prévision du modèle",
                            paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(f, width="stretch")
        cal = load_calendar(); cal = upcoming(cal[cal.iso3 == iso])
        st.subheader("Prochains scrutins")
        if len(cal):
            st.dataframe(cal[["date", "election", "statut", "source"]], hide_index=True, width="stretch",
                         column_config={"date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"), "source": st.column_config.LinkColumn("Source")})
        else:
            st.info("Aucun scrutin enregistré pour ce pays. Ajoutez-le dans la page Élections & risque politique.")
        polls = load_polls()
        agg = aggregate_polls(polls, iso)
        if agg:
            st.subheader(f"Sondages agrégés ({agg['n_polls']} sondages, dernier le {agg['last_poll']:%d/%m/%Y})")
            st.dataframe(agg["table"], width="stretch")
        st.subheader("Simulateur : et si…")
        row = store.pillars.loc[iso] if iso in store.pillars.index else None
        if row is not None:
            cols = st.columns(4)
            changes = {}
            for i, a in enumerate(AGENTS):
                changes[a["id"]] = cols[i % 4].slider(a["nom"], -30, 30, 0, 5, key=f"wi{a['id']}")
            new = what_if(row, changes)
            old = what_if(row, {})
            st.markdown(kpi_html([(fnum(old, 0), "Situation actuelle", color_class(old)),
                                  (fnum(new, 0), "Situation simulée", color_class(new)),
                                  (f"{'+' if new - old >= 0 else ''}{fnum(new - old, 1)}", "Écart (points)", "")]), unsafe_allow_html=True)

    with tabs[4]:
        from core.markets import TICKERS, fetch_market
        if iso in TICKERS:
            mk = st.session_state.get(f"mk_{iso}")
            if mk is None:
                with st.spinner("Lecture des cours de bourse…"):
                    mk = fetch_market(iso)
                st.session_state[f"mk_{iso}"] = mk
            if mk and mk.get("dernier"):
                st.markdown(kpi_html([(fnum(mk["dernier"], 0), mk["nom"], ""), (f"{fnum(mk['perf_1a'], 1)} %", "Sur 1 an", "vert" if mk["perf_1a"] > 0 else "rouge"),
                                      (f"{fnum(mk['perf_5a'], 1)} %", "Sur 5 ans", "vert" if (mk["perf_5a"] or 0) > 0 else "rouge"),
                                      (f"{fnum(mk['volatilite_1a'], 1)} %", "Volatilité annualisée", ""),
                                      (f"{fnum(mk['drawdown_max'], 0)} %", "Pire baisse (10 ans)", "")]), unsafe_allow_html=True)
                f = px.line(mk["serie"], height=320, labels={"value": "", "index": ""})
                f.update_traces(line_color="#1F5C7A"); f.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(f, width="stretch")
                st.caption(f"Source : {mk['source']} · dernière cotation {mk['date']}" + (" · données en cache (hors ligne)" if mk.get("hors_ligne") else ""))
            else:
                st.info("Cours indisponibles pour le moment (Yahoo Finance injoignable). Les indicateurs structurels ci-dessous restent valables.")
        else:
            st.info("Pas d'indice boursier national suivi pour ce pays ; voici les indicateurs financiers structurels.")
        p = an["piliers"]["finance"]
        st.markdown("<div class='wh-prose'>" + "".join(f"<p>{t}</p>" for t in p["texte"]) + "</div>", unsafe_allow_html=True)
        e = an["piliers"]["economie"]["indicateurs"]
        if len(e):
            st.dataframe(pd.DataFrame({"Indicateur économique": e.label, "Valeur": [fval(c_, v) for c_, v in zip(e.code, e.value)], "Année": e.year.astype(int)}),
                         hide_index=True, width="stretch")

    with tabs[5]:
        from core.news import collect, load_cached, clean_title
        nkey = f"news_{iso}"
        if st.button("Lire les dernières actualités", key="nbtn"):
            with st.spinner("Collecte des titres de presse…"):
                st.session_state[nkey] = collect(iso, an["nom"], c["name_en"])
        news = st.session_state.get(nkey) or load_cached(iso)
        if news.get("ok") and news.get("n"):
            st.markdown(kpi_html([(news["n"], "Titres analysés", ""), (fnum(news["tonalite_moyenne"], 2), "Tonalité moyenne", "vert" if news["tonalite_moyenne"] > 0 else "rouge"),
                                  (news["n_declarations"], "Déclarations politiques", ""), (news["n_signaux_impunite"], "Signaux corruption / impunité", "orange" if news["n_signaux_impunite"] else "")]),
                        unsafe_allow_html=True)
            t = news["table"].head(30)
            st.dataframe(pd.DataFrame({"Date": t["date"], "Titre": t["titre"].map(clean_title), "Média": t["media"], "Lien": t["lien"],
                                       "Déclaration": t["declaration"], "Impunité/corruption": t["signal_impunite"]}),
                         hide_index=True, width="stretch", column_config={"Lien": st.column_config.LinkColumn("Lien", display_text="ouvrir")})
            if news.get("gdelt_tone") is not None:
                st.caption(f"Tonalité GDELT (presse mondiale anglophone, 3 mois) : {fnum(news['gdelt_tone'], 2)}")
        else:
            st.info("Aucune actualité en cache pour ce pays. Cliquez sur « Lire les dernières actualités » (Google News et GDELT).")

    with tabs[6]:
        from core import llm
        from core.news import load_cached, wikipedia_summary
        prov = llm.provider()
        st.markdown("Le texte ci-dessous est rédigé à partir des seules données des agents, des actualités collectées, de Wikipédia et "
                    "des références d'experts (FMI, Banque mondiale, OCDE, V-Dem, Transparency International, GIEC…).")
        if not prov:
            st.info("Pour activer l'avis d'expert rédigé par IA, ajoutez une clé dans les secrets : ANTHROPIC_API_KEY, MISTRAL_API_KEY, "
                    "ou OLLAMA_URL pour un modèle local. L'analyse complète des autres onglets fonctionne sans clé.")
        elif st.button(f"Rédiger l'avis d'expert ({prov})", type="primary"):
            with st.spinner("Rédaction en cours (30 à 90 secondes)…"):
                try:
                    txt = llm.expert_analysis(an, st.session_state.get(f"news_{iso}") or load_cached(iso), wikipedia_summary(an["nom"]))
                    st.session_state[f"llm_{iso}"] = txt
                except Exception as e:
                    st.error(f"Le modèle n'a pas répondu : {e}. Vérifiez la clé et le nom du modèle dans les secrets.")
        if st.session_state.get(f"llm_{iso}"):
            st.markdown(st.session_state[f"llm_{iso}"])

    with tabs[7]:
        from core import pdf_reports as pr
        from core.markets import TICKERS
        from core.news import load_cached
        st.markdown("Chaque rapport cite l'origine de chaque donnée (indicateur, code, producteur, année).")
        a, b, c3 = st.columns(3)
        with a:
            st.subheader("Rapport global")
            if st.button("Générer le rapport global", key="pdfg", width="stretch"):
                with st.spinner("Mise en page du rapport…"):
                    cal = upcoming(load_calendar()); cal = cal[cal.iso3 == iso]
                    st.session_state[f"pdfg_{iso}"] = pr.country_report(store, an, news=st.session_state.get(f"news_{iso}") or load_cached(iso),
                                                                        market=st.session_state.get(f"mk_{iso}"), expert_text=st.session_state.get(f"llm_{iso}"),
                                                                        calendar=cal, polls=aggregate_polls(load_polls(), iso))
            if st.session_state.get(f"pdfg_{iso}"):
                st.download_button("Télécharger le PDF", st.session_state[f"pdfg_{iso}"], f"WorldHorizon_{iso}_rapport_global.pdf", "application/pdf", width="stretch")
        with b:
            st.subheader("Rapport par critère")
            nom2id = {x["nom"]: x["id"] for x in AGENTS}
            pid = nom2id[st.selectbox("Critère", list(nom2id), key="pdfc_sel")]
            if st.button("Générer le rapport du critère", key="pdfc", width="stretch"):
                with st.spinner("Mise en page…"):
                    st.session_state[f"pdfc_{iso}_{pid}"] = pr.criterion_report(store, an, pid)
            if st.session_state.get(f"pdfc_{iso}_{pid}"):
                st.download_button("Télécharger le PDF", st.session_state[f"pdfc_{iso}_{pid}"], f"WorldHorizon_{iso}_{pid}.pdf", "application/pdf", width="stretch")
        with c3:
            st.subheader("Élections & risque")
            if st.button("Générer le rapport élections", key="pdfe", width="stretch"):
                cal = upcoming(load_calendar()); cal = cal[cal.iso3 == iso]
                st.session_state[f"pdfe_{iso}"] = pr.elections_report(store, an, cal, aggregate_polls(load_polls(), iso),
                                                                      st.session_state.get(f"news_{iso}") or load_cached(iso))
            if st.session_state.get(f"pdfe_{iso}"):
                st.download_button("Télécharger le PDF", st.session_state[f"pdfe_{iso}"], f"WorldHorizon_{iso}_elections.pdf", "application/pdf", width="stretch")
        st.divider()
        st.subheader("Images")
        pil = store.pillars.loc[iso].to_dict() if iso in store.pillars.index else {}
        g = lambda k: float(m.get(k)) if pd.notna(m.get(k)) else float("nan")  # noqa: E731
        poster = im.country_poster(iso, an["nom"], g("score_global"), g("score_actuel"), g("indice_avenir"), pil, f"{c['region_fr']} · {c['income_fr']}")
        st.image(poster, width="stretch")
        i1, i2, i3 = st.columns(3)
        i1.download_button("Télécharger l'affiche (PNG)", poster, f"WorldHorizon_{iso}_affiche.png", "image/png", width="stretch")
        i2.download_button("Télécharger le radar (PNG)", im.radar(pil, an["nom"], compare=store.pillars.mean().to_dict()),
                           f"WorldHorizon_{iso}_radar.png", "image/png", width="stretch")
        i3.download_button("Télécharger la carte (PNG)", im.world_map(M["score_global"], highlight=iso, title=f"Score global — {an['nom']} en surbrillance"),
                           f"WorldHorizon_{iso}_carte.png", "image/png", width="stretch")

    with tabs[8]:
        st.dataframe(an["sources"], hide_index=True, width="stretch", column_config={"lien": st.column_config.LinkColumn("Lien")})
        st.markdown("**Analyses d'experts de référence :**")
        for a in AGENTS:
            st.markdown(f"- {a['nom']} : " + " · ".join(f"[{n}]({u})" for n, u in an["piliers"][a["id"]]["experts"]))

    with tabs[9]:
        from core import intel_eco as ie
        ie.country_panel(store, iso, key="pays")
        with st.expander("⛓️ Dépendances sur 30 ans"):
            ie.dependencies_panel(store, iso, key="pays")
        st.caption("Carte mondiale, comparaisons, grilles d'analyse et bibliothèque de sources : page « Intelligence économique ».")


def page_intel_eco():
    from core import intel_eco as ie
    ie.page(store, ISO_LIST, LABELS)


def page_compare():
    st.markdown("<div class='wh-hero'><h1 class='wh-title'>Comparer des pays</h1></div>", unsafe_allow_html=True)
    default = [x for x in [st.session_state.get("iso"), "FRA", "SEN", "HTI"] if x][:3]
    sel = st.multiselect("Pays (jusqu'à 6)", ISO_LIST, default=list(dict.fromkeys(default)), format_func=LABELS.get, max_selections=6)
    if not sel:
        return
    st.plotly_chart(_radar_fig({store.name(i): store.pillars.loc[i].to_dict() for i in sel if i in store.pillars.index}, 480), width="stretch")
    cols = ["name_fr", "score_global", "score_actuel", "indice_avenir", "risque_politique"] + [f"p_{a['id']}" for a in AGENTS]
    t = M.loc[[i for i in sel if i in M.index], cols].round(0)
    t.columns = ["Pays", "Global", "Actuel", "Avenir 2100", "Risque politique"] + [a["nom"] for a in AGENTS]
    st.dataframe(t.set_index("Pays").T, width="stretch")
    h = store.global_hist[store.global_hist.iso3.isin(sel)].copy()
    h["Pays"] = h.iso3.map(store.name)
    f = px.line(h, x="year", y="score_actuel", color="Pays", height=360, labels={"year": "", "score_actuel": "score de situation"})
    f.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 100])
    st.plotly_chart(f, width="stretch")


def page_elections():
    st.markdown("<div class='wh-hero'><h1 class='wh-title'>Élections & risque politique</h1>"
                "<p>Le modèle estime la probabilité d'un recul démocratique marqué dans les trois ans ; l'indice combine "
                "cette probabilité avec la fragilité structurelle (impunité, corruption, sécurité, économie, social).</p></div>", unsafe_allow_html=True)
    ev = st.plotly_chart(world_map("risque_politique", height=470), width="stretch", on_select="rerun",
                         selection_mode="points", key="map_risk", config={"displayModeBar": False})
    iso = map_click_iso(ev)
    if iso and iso in ISO_LIST:
        go_country(iso)
    met = store.ml_metrics.tail(1)
    if len(met):
        r = met.iloc[0]
        st.caption(f"Modèle : {r['model']} · cible : {r['target']} · {int(r['n_train'])} observations d'entraînement · "
                   f"AUC hors échantillon {r['auc_test']:.2f} · fréquence de base {r['base_rate'] * 100:.0f} %")
    t1, t2, t3, t4 = st.tabs(["Calendrier", "Sondages", "Classement du risque", "Modèle"])
    with t1:
        cal = load_calendar().reset_index(drop=True)
        cal["pays"] = cal["iso3"].map(store.name)
        edited = st.data_editor(cal[["iso3", "pays", "date", "election", "type", "statut", "source"]], num_rows="dynamic",
                                width="stretch", hide_index=True, disabled=["pays"],
                                column_config={"date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                                               "iso3": st.column_config.SelectboxColumn("Code pays", options=ISO_LIST),
                                               "source": st.column_config.LinkColumn("Source")})
        if st.button("Enregistrer le calendrier"):
            out = edited.drop(columns=["pays"]).copy()
            out["date"] = pd.to_datetime(out["date"]).dt.date
            out.to_csv(CAL, index=False)
            st.success("Calendrier enregistré.")
        st.caption("Les dates marquées « à confirmer » sont indicatives. Le bouton de mise à jour (option calendrier) complète la liste depuis Wikipédia.")
    with t2:
        polls = load_polls()
        st.markdown("Saisissez ou importez des sondages : une ligne par candidat (ou parti) et par sondage.")
        up = st.file_uploader("Importer un CSV de sondages (mêmes colonnes)", type="csv")
        if up is not None:
            polls = pd.concat([polls, pd.read_csv(up, parse_dates=["date_sondage"])], ignore_index=True)
        ed = st.data_editor(polls.reset_index(drop=True), num_rows="dynamic", width="stretch", hide_index=True,
                            column_config={"date_sondage": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                                           "iso3": st.column_config.SelectboxColumn("Code pays", options=ISO_LIST)})
        if st.button("Enregistrer les sondages"):
            ed.to_csv(POLLS, index=False); st.success("Sondages enregistrés.")
        if len(ed.dropna(subset=["iso3"])):
            pays = st.selectbox("Pays à agréger", sorted(ed["iso3"].dropna().unique()), format_func=LABELS.get)
            el = st.selectbox("Élection", sorted(ed[ed.iso3 == pays]["election"].dropna().unique()))
            ed["date_sondage"] = pd.to_datetime(ed["date_sondage"])
            agg = aggregate_polls(ed, pays, el)
            if agg:
                st.dataframe(agg["table"], width="stretch")
                f = px.bar(agg["table"].reset_index(), x="candidat_ou_parti", y="proba_en_tete", height=300,
                           labels={"candidat_ou_parti": "", "proba_en_tete": "probabilité d'arriver en tête"})
                f.update_traces(marker_color="#1F5C7A"); f.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_tickformat=".0%")
                st.plotly_chart(f, width="stretch")
    with t3:
        r = store.risk.join(M[["name_fr", "score_global"]]).dropna(subset=["indice_risque_politique"]).sort_values("indice_risque_politique", ascending=False)
        st.dataframe(r[["name_fr", "indice_risque_politique", "niveau", "proba_recul_democratique_3ans", "fragilite_structurelle", "polarisation_centile"]].round(2),
                     hide_index=True, width="stretch",
                     column_config={"name_fr": "Pays", "indice_risque_politique": "Indice /100", "niveau": "Niveau",
                                    "proba_recul_democratique_3ans": st.column_config.ProgressColumn("Probabilité de recul (3 ans)", min_value=0, max_value=1, format="%.2f"),
                                    "fragilite_structurelle": "Fragilité", "polarisation_centile": "Polarisation (centile)"})
    with t4:
        imp = store.importance.copy()
        if len(imp):
            from core.ml_engine import FEATURE_LABELS
            imp.columns = ["facteur", "importance"]
            imp["facteur"] = imp["facteur"].map(lambda k: FEATURE_LABELS.get(k, k))
            f = px.bar(imp.head(12).iloc[::-1], x="importance", y="facteur", orientation="h", height=380,
                       labels={"importance": "perte d'AUC si la variable est brouillée", "facteur": ""})
            f.update_traces(marker_color="#1F5C7A"); f.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(f, width="stretch")
        hist = store.ml_metrics
        if len(hist) > 1:
            st.markdown("**Apprentissage au fil des mises à jour**")
            st.line_chart(hist.set_index("run_at")[["auc_test"]])


def page_future():
    st.markdown("<div class='wh-hero'><h1 class='wh-title'>Prospective 2050–2100</h1>"
                "<p>L'indice d'avenir mesure les fondations d'un siècle prospère : démographie (projections ONU), capital humain, "
                "institutions, climat et ressources, dynamique économique. Ce sont des scénarios, pas des prédictions.</p></div>", unsafe_allow_html=True)
    d = M.dropna(subset=["score_actuel", "indice_avenir"]).reset_index()
    f = px.scatter(d, x="score_actuel", y="indice_avenir", hover_name="name_fr", color="couleur", color_discrete_map=HEX,
                   height=520, labels={"score_actuel": "situation actuelle", "indice_avenir": "indice d'avenir 2100"})
    f.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="#9AA5AE", dash="dot"))
    f.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(f, width="stretch")
    st.caption("Au-dessus de la diagonale : pays dont les fondations d'avenir dépassent la situation actuelle (potentiel de progression).")
    d["potentiel"] = d["indice_avenir"] - d["score_actuel"]
    a, b = st.columns(2)
    a.subheader("Plus fort potentiel de progression")
    a.dataframe(d.nlargest(12, "potentiel")[["name_fr", "score_actuel", "indice_avenir", "score_2050_central", "var_pop_2100_pct"]].round(0),
                hide_index=True, width="stretch",
                column_config={"name_fr": "Pays", "score_actuel": "Actuel", "indice_avenir": "Avenir", "score_2050_central": "2050 (central)", "var_pop_2100_pct": "Population 2100 (%)"})
    b.subheader("Meilleures fondations pour 2100")
    b.dataframe(d.nlargest(12, "indice_avenir")[["name_fr", "score_actuel", "indice_avenir", "score_2050_central", "var_pop_2100_pct"]].round(0),
                hide_index=True, width="stretch",
                column_config={"name_fr": "Pays", "score_actuel": "Actuel", "indice_avenir": "Avenir", "score_2050_central": "2050 (central)", "var_pop_2100_pct": "Population 2100 (%)"})


def page_rankings():
    st.markdown("<div class='wh-hero'><h1 class='wh-title'>Classements</h1></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cont = c1.multiselect("Continent", sorted(M["continent_fr"].dropna().unique()))
    coul = c2.multiselect("Couleur", ["vert", "orange", "rouge", "gris"])
    terr = c3.selectbox("Périmètre", ["Pays et territoires", "Pays seulement", "Territoires seulement"])
    d = M.copy()
    if cont: d = d[d.continent_fr.isin(cont)]
    if coul: d = d[d.couleur.isin(coul)]
    if terr == "Pays seulement": d = d[d.territoire.astype(str) != "True"]
    if terr == "Territoires seulement": d = d[d.territoire.astype(str) == "True"]
    cols = ["name_fr", "rang", "score_global", "score_actuel", "indice_avenir", "risque_politique", "fiabilite"] + [f"p_{a['id']}" for a in AGENTS]
    show = d[cols].sort_values("score_global", ascending=False).round(0)
    cfg = {"name_fr": "Pays", "rang": "Rang", "score_global": "Global", "score_actuel": "Actuel", "indice_avenir": "Avenir",
           "risque_politique": "Risque", "fiabilite": "Fiabilité %", **{f"p_{a['id']}": a["nom"] for a in AGENTS}}
    st.dataframe(show, hide_index=True, width="stretch", height=620, column_config=cfg)
    st.download_button("Télécharger le classement (CSV)", show.to_csv().encode("utf-8"), "worldhorizon_classement.csv", "text/csv")


def page_data():
    st.markdown("<div class='wh-hero'><h1 class='wh-title'>Données, agents et méthode</h1></div>", unsafe_allow_html=True)
    st.subheader("Mettre à jour")
    st.markdown("Les agents relancent la collecte, les CSV historiques sont complétés (rien n'est effacé), les modèles de "
                "machine learning sont réentraînés et la carte est recalculée.")
    update_panel(compact=False)
    log = AGENTS_DIR / "_runs_log.csv"
    if log.exists():
        st.subheader("Journal des collectes")
        st.dataframe(pd.read_csv(log).iloc[::-1].head(60), hide_index=True, width="stretch")
    st.subheader("Fichiers CSV")
    files = sorted([p for p in (ROOT / "data").rglob("*.csv")], key=lambda p: str(p))
    f = st.selectbox("Fichier", files, format_func=lambda p: str(p.relative_to(ROOT)))
    if f:
        df = pd.read_csv(f)
        st.caption(f"{len(df):,} lignes · {f.stat().st_size / 1e6:.1f} Mo".replace(",", " "))
        st.dataframe(df.head(500), width="stretch")
        st.download_button("Télécharger ce CSV", f.read_bytes(), f.name, "text/csv")
    st.subheader("Méthode")
    st.markdown((ROOT / "docs" / "METHODOLOGIE.md").read_text() if (ROOT / "docs" / "METHODOLOGIE.md").exists() else "")


PAGES = {
    "home": st.Page(page_home, title="Carte du monde", icon="🗺️", default=True),
    "pays": st.Page(page_country, title="Analyse d'un pays", icon="🔎", url_path="pays"),
    "compare": st.Page(page_compare, title="Comparer", icon="⚖️", url_path="comparer"),
    "elections": st.Page(page_elections, title="Élections & risque", icon="🗳️", url_path="elections"),
    "ie": st.Page(page_intel_eco, title="Intelligence économique", icon="🕵️", url_path="intelligence-economique"),
    "future": st.Page(page_future, title="Prospective 2100", icon="🔭", url_path="prospective"),
    "rank": st.Page(page_rankings, title="Classements", icon="📊", url_path="classements"),
    "data": st.Page(page_data, title="Données & méthode", icon="🗄️", url_path="donnees"),
}
FUNCS = {"home": page_home, "pays": page_country, "compare": page_compare, "elections": page_elections,
         "ie": page_intel_eco, "future": page_future,"rank": page_rankings, "data": page_data}
nav = st.navigation(list(PAGES.values()))
sidebar()
if st.session_state.get("_page") in FUNCS:   # utilisé par les tests automatiques (streamlit.testing)
    FUNCS[st.session_state["_page"]]()
else:
    nav.run()
