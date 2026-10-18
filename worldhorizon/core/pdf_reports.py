"""Moteur PDF (ReportLab) : rapport global pays, rapport détaillé par critère, rapport élections.
Chaque rapport cite ses sources (indicateur, code, producteur, année) et dépasse 2 pages."""
from __future__ import annotations

import datetime as dt
import io
import re
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from . import images as im
from .catalog import AGENTS, AGENT_BY_ID, COLOR_THRESHOLDS
from .narrative import COLOR_LABEL, fnum, fval, ind_def, per_capita_series

_FD = matplotlib.get_data_path() + "/fonts/ttf/"
pdfmetrics.registerFont(TTFont("DV", _FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", _FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVI", _FD + "DejaVuSans-Oblique.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402

registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

INK = colors.HexColor("#0f1b2d"); MUTED = colors.HexColor("#5b6b82"); LINE = colors.HexColor("#dfe5ee")
ACC = colors.HexColor("#2f6fde"); SOFT = colors.HexColor("#f3f6fb")
COL = {k: colors.HexColor(v) for k, v in im.PALETTE.items()}

S = {
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=20, leading=24, textColor=INK, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=13.5, leading=17, textColor=INK, spaceBefore=10, spaceAfter=5),
    "h3": ParagraphStyle("h3", fontName="DVB", fontSize=10.5, leading=13, textColor=ACC, spaceBefore=6, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="DV", fontSize=9.2, leading=13.2, textColor=INK, spaceAfter=4, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="DV", fontSize=7.4, leading=9.6, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.6, leading=9.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=7.6, leading=9.6, textColor=INK),
    "kpi": ParagraphStyle("kpi", fontName="DVB", fontSize=22, leading=24, textColor=INK),
    "kpil": ParagraphStyle("kpil", fontName="DV", fontSize=7.5, leading=9, textColor=MUTED),
}


def P(t, st="p"):
    return Paragraph(_md(escape(str(t))), S[st])


def _md(t: str) -> str:
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", t)
    return t


def IMG(png: bytes, width_cm: float):
    from reportlab.lib.utils import ImageReader
    r = ImageReader(io.BytesIO(png)); w, h = r.getSize()
    return Image(io.BytesIO(png), width=width_cm * cm, height=width_cm * cm * h / w)


def _table(rows, widths, header=True, zebra=True, font=7.6):
    data = [[c if not isinstance(c, str) else Paragraph(escape(c), S["cellb" if (header and i == 0) else "cell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), SOFT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK)]
    t.setStyle(TableStyle(style))
    return t


def _dot(score):
    c = im.score_color(score)
    return f'<font color="{c}">●</font>'


class _Doc:
    def __init__(self, title: str, subtitle: str):
        self.buf = io.BytesIO(); self.title = title; self.subtitle = subtitle
        self.doc = SimpleDocTemplate(self.buf, pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm,
                                     topMargin=1.8 * cm, bottomMargin=1.6 * cm, title=title, author="WorldHorizon")

    def _decor(self, canv, doc):
        canv.saveState()
        canv.setFont("DVB", 7.5); canv.setFillColor(MUTED)
        canv.drawString(1.7 * cm, A4[1] - 1.1 * cm, "WORLDHORIZON")
        canv.setFont("DV", 7.5)
        canv.drawRightString(A4[0] - 1.7 * cm, A4[1] - 1.1 * cm, self.title)
        canv.setStrokeColor(LINE); canv.line(1.7 * cm, A4[1] - 1.25 * cm, A4[0] - 1.7 * cm, A4[1] - 1.25 * cm)
        canv.drawString(1.7 * cm, 0.9 * cm, f"Généré le {dt.date.today().strftime('%d/%m/%Y')} · données publiques : "
                                            "Banque mondiale, V-Dem, ONU, OCDE, Transparency International")
        canv.drawRightString(A4[0] - 1.7 * cm, 0.9 * cm, f"page {doc.page}")
        canv.restoreState()

    def build(self, story) -> bytes:
        self.doc.build(story, onFirstPage=self._decor, onLaterPages=self._decor)
        return self.buf.getvalue()


def _kpis(m) -> Table:
    def cell(v, lab, color=None):
        txt = fnum(v, 0) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else (v if isinstance(v, str) else "n.d.")
        size = 22 if len(str(txt)) <= 4 else 13
        st = ParagraphStyle("k", parent=S["kpi"], fontSize=size, leading=size + 2, textColor=colors.HexColor(color) if color else INK)
        return [Paragraph(escape(str(txt)), st), Paragraph(escape(lab), S["kpil"])]
    rang = f"{int(m['rang'])}e/{int(m['n_classes'])}" if pd.notna(m.get("rang")) else "n.d."
    cells = [cell(m.get("score_global"), "Score global /100", im.score_color(m.get("score_global"))),
             cell(m.get("score_actuel"), "Situation actuelle"), cell(m.get("indice_avenir"), "Indice d'avenir 2100"),
             cell(rang, "Rang mondial"), cell(str(m.get("niveau_risque", "n.d.")), "Risque politique")]
    t = Table([[c[0] for c in cells], [c[1] for c in cells]], colWidths=[3.5 * cm] * 5)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("LINEAFTER", (0, 0), (-2, -1), 0.5, colors.white),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 1), (-1, 1), 7)]))
    return t


def _pillar_table(st, an):
    rows = [["Pilier", "Score", "Niveau", "Rang", "Moy. région", "Tendance 10 ans"]]
    for a in AGENTS:
        p = an["piliers"][a["id"]]
        h = p["historique"].dropna()
        d = ""
        if len(h) > 10 and len(h[h.index <= h.index.max() - 10]):
            dv = h.iloc[-1] - h[h.index <= h.index.max() - 10].iloc[-1]
            d = f"{'+' if dv >= 0 else ''}{fnum(dv, 1)} pts"
        rows.append([Paragraph(f"{_dot(p['score'])} {escape(p['nom'])}", S["cell"]), fnum(p["score"], 0), p["niveau"],
                     f"{p['rang']}/{p['n']}" if p["rang"] else "—", fnum(p["reg_mean"], 0), d])
    return _table(rows, [5.4, 1.5, 2.6, 1.8, 2.1, 2.6])


def _indicator_table(p):
    rows = [["Indicateur", "Valeur", "Année", "Médiane région", "Médiane monde", "Score"]]
    for _, r in p["indicateurs"].iterrows():
        rows.append([r.label, fval(r.code, r.value), str(int(r.year)), fval(r.code, r.reg_median),
                     fval(r.code, r.world_median), fnum(r.score, 0)])
    return _table(rows, [5.4, 2.7, 1.3, 2.6, 2.6, 1.3])


def _sources(an, only_pillar: str | None = None):
    src = an["sources"]
    if only_pillar:
        src = src[src.pilier == AGENT_BY_ID[only_pillar]["nom"]]
    rows = [["Pilier", "Indicateur (code)", "Année", "Producteur de la donnée", "Lien"]]
    for _, r in src.iterrows():
        rows.append([r.pilier, f"{r.indicateur} ({r.code})", str(r.annee), r.source, r.lien])
    return _table(rows, [2.6, 4.7, 1.3, 4.4, 4.6])


def _experts(an, pids):
    out = []
    for pid in pids:
        p = an["piliers"][pid]
        for n, u in p["experts"]:
            out.append(Paragraph(f"• <b>{escape(n)}</b> — <link href='{escape(u)}' color='#2f6fde'>{escape(u)}</link>", S["small"]))
    return out


def _methodo():
    return [P("Méthodologie", "h3"), P(
        "Chaque agent collecte ses séries de façon indépendante (Banque mondiale WDI, V-Dem v16, Gapminder — ONU WPP, "
        "IDEA, EIU, Transparency International, World Happiness Report — et OCDE PISA 2022). Les valeurs les plus récentes "
        "(≤ 8 ans) sont normalisées de 0 à 100 entre le 5e et le 95e centile mondial, dans le sens favorable. "
        "Un pilier est noté si au moins 35 % de ses indicateurs pondérés sont disponibles. Score actuel = moyenne pondérée "
        "des 14 piliers ; score global = 75 % situation actuelle + 25 % indice d'avenir (démographie ONU 2100, capital humain, "
        f"institutions, climat et ressources, dynamique économique). Couleurs : vert ≥ {COLOR_THRESHOLDS['vert']}, "
        f"orange ≥ {COLOR_THRESHOLDS['orange']}, rouge en dessous. Les projections prolongent les tendances de 25 ans "
        "avec un amortissement (demi-vie 15 ans) ; ce sont des scénarios, pas des certitudes.", "small")]


# --------------------------------------------------------------------------- rapport pays
def country_report(st, an: dict, news: dict | None = None, market: dict | None = None,
                   expert_text: str | None = None, calendar: pd.DataFrame | None = None, polls: dict | None = None) -> bytes:
    iso, name, m, c = an["iso3"], an["nom"], an["master"], an["pays"]
    D = _Doc(f"Rapport global — {name}", "")
    pil = st.pillars.loc[iso].to_dict() if iso in st.pillars.index else {}
    sc = lambda k: float(m.get(k)) if pd.notna(m.get(k)) else float("nan")  # noqa: E731
    story = [IMG(im.country_poster(iso, name, sc("score_global"), sc("score_actuel"), sc("indice_avenir"), pil,
                                   f"{c['region_fr']} · {c['income_fr']}"), 17.6), Spacer(1, 8),
             P(f"Rapport global — {name}", "h1"),
             P(f"{COLOR_LABEL.get(m.get('couleur', 'gris')).capitalize()} · {c['continent_fr']} · capitale : {c.get('capital') or 'n.d.'} · "
               f"monnaie : {c.get('currency') or 'n.d.'}", "small"), Spacer(1, 6), _kpis(m), Spacer(1, 8),
             P("Synthèse exécutive", "h2")]
    story += [P(t) for t in an["synthese"]]
    story += [PageBreak(), P("Profil sur les 14 piliers", "h2")]
    world = st.pillars.mean().to_dict()
    row = Table([[IMG(im.radar(pil, "", compare=world), 8.2), IMG(im.pillar_bars(pil), 9.2)]], colWidths=[8.4 * cm, 9.4 * cm])
    story += [row, Spacer(1, 4), _pillar_table(st, an), Spacer(1, 6)]
    sw = an["swot"]
    swt = Table([[Paragraph("<b>Forces</b><br/>" + "<br/>".join("• " + escape(x) for x in sw["Forces"]), S["cell"]),
                  Paragraph("<b>Faiblesses</b><br/>" + "<br/>".join("• " + escape(x) for x in sw["Faiblesses"]), S["cell"])],
                 [Paragraph("<b>Opportunités</b><br/>" + "<br/>".join("• " + escape(x) for x in sw["Opportunités"]), S["cell"]),
                  Paragraph("<b>Menaces</b><br/>" + "<br/>".join("• " + escape(x) for x in sw["Menaces"]), S["cell"])]],
                colWidths=[8.8 * cm, 8.8 * cm])
    swt.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#e8f6f0")), ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#fdecec")),
                             ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#eaf1fd")), ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#fff4e5")),
                             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [P("Matrice SWOT", "h2"), swt]

    story += [PageBreak(), P("Situation détaillée, domaine par domaine", "h2")]
    for a in AGENTS:
        p = an["piliers"][a["id"]]
        block = [P(f"{p['nom']} — {fnum(p['score'], 0)}/100", "h3")] + [P(t) for t in p["texte"]]
        story += [KeepTogether(block[:3])] + block[3:] + [Spacer(1, 3)]

    story += [PageBreak(), P("Prospective : 2050 et horizon 2100", "h2")]
    proj = st.projections[st.projections.iso3 == iso]
    story.append(IMG(im.history_projection(an["historique_global"], proj), 17))
    pop = st.series(iso, "pop")
    if len(pop):
        story.append(IMG(im.population_chart(pop, name), 17))
    if an["avenir"]["composantes"]:
        story.append(IMG(im.future_components(an["avenir"]["composantes"]), 15))
    story += [P(t) for t in an["avenir"]["texte"]]

    story += [P("Politique, élections et risque", "h2")] + [P(t) for t in an["politique"]["texte"]]
    if an["politique"]["facteurs"]:
        rows = [["Facteur (modèle ML)", "Effet sur la probabilité de recul démocratique"]]
        rows += [[k, f"{'+' if v >= 0 else ''}{fnum(v * 100, 1)} pt(s) de probabilité"] for k, v in an["politique"]["facteurs"]]
        story.append(_table(rows, [7, 8]))
    if calendar is not None and len(calendar):
        rows = [["Date", "Scrutin", "Statut"]] + [[d.strftime("%d/%m/%Y"), e, s] for d, e, s in
                                                  zip(calendar["date"], calendar["election"], calendar["statut"])]
        story += [P("Prochains scrutins", "h3"), _table(rows, [2.6, 9, 6])]
    if polls:
        t = polls["table"]
        rows = [["Candidat / parti", "Moyenne pondérée", "Intervalle 80 %", "Probabilité d'arriver en tête"]]
        rows += [[i, f"{r.moyenne_ponderee:.1f} %", f"{r.intervalle_80_bas:.0f}–{r.intervalle_80_haut:.0f} %", f"{r.proba_en_tete * 100:.0f} %"] for i, r in t.iterrows()]
        story += [P(f"Agrégation des sondages ({polls['n_polls']} sondages, dernier le {polls['last_poll']})", "h3"), _table(rows, [6, 3.5, 3.5, 4.6])]

    story += [P("Marchés financiers", "h2")]
    if market and market.get("dernier"):
        rows = [["Indice", "Dernier", "Date", "1 mois", "1 an", "5 ans", "Volatilité 1 an", "Drawdown max"],
                [market["nom"], fnum(market["dernier"], 0), market["date"], f"{fnum(market['perf_1m'], 1)} %",
                 f"{fnum(market['perf_1a'], 1)} %", f"{fnum(market['perf_5a'], 1)} %", f"{fnum(market['volatilite_1a'], 1)} %",
                 f"{fnum(market['drawdown_max'], 0)} %"]]
        story += [_table(rows, [3.4, 1.8, 2, 1.5, 1.5, 1.6, 2.4, 2.4]), P(f"Source : {market['source']}", "small")]
    story += [P(t) for t in an["piliers"]["finance"]["texte"][:3]]

    if news and news.get("ok") and news.get("n"):
        story += [P("Veille : actualités et déclarations récentes", "h2"),
                  P(f"{news['n']} titres analysés · tonalité moyenne {fnum(news['tonalite_moyenne'], 2)} · "
                    f"{news['n_declarations']} déclarations politiques · {news['n_signaux_impunite']} signaux liés à la corruption ou à l'impunité.", "small")]
        for _, r in news["table"].head(12).iterrows():
            d = r["date"].strftime("%d/%m/%Y") if pd.notna(r["date"]) else ""
            story.append(Paragraph(f"• {escape(d)} — {escape(str(r['titre']))}", S["small"]))
    if expert_text:
        story += [PageBreak(), P("Avis d'expert (rédigé par modèle de langage à partir des données ci-dessus)", "h2")]
        for line in expert_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                story.append(P(line.lstrip("# "), "h3"))
            else:
                story.append(P(line.lstrip("-• ")))
    story += [PageBreak(), P("Sources et traçabilité", "h2"), _sources(an), Spacer(1, 6),
              P("Analyses d'experts de référence pour ce pays", "h3")] + _experts(an, [a["id"] for a in AGENTS]) + _methodo()
    return D.build(story)


# --------------------------------------------------------------------------- rapport par critère
def criterion_report(st, an: dict, pid: str) -> bytes:
    iso, name = an["iso3"], an["nom"]
    p = an["piliers"][pid]
    D = _Doc(f"{p['nom']} — {name}", "")
    story = [P(f"{p['nom']} — {name}", "h1"), P(p["description"], "small"), Spacer(1, 6)]
    def kc(txt, color=None):
        size = 22 if len(str(txt)) <= 4 else 13
        return Paragraph(f"<font color='{color}'>{escape(str(txt))}</font>" if color else escape(str(txt)),
                         ParagraphStyle("kc", parent=S["kpi"], fontSize=size, leading=size + 2))
    k = Table([[kc(fnum(p["score"], 0), im.score_color(p["score"])), kc(p["niveau"]), kc(f"{p['rang'] or '—'}/{p['n']}"),
                kc(fnum(p["reg_mean"], 0)), kc(fnum(p["world_mean"], 0))],
               [Paragraph(x, S["kpil"]) for x in ("Score /100", "Niveau", "Rang mondial", "Moyenne régionale", "Moyenne mondiale")]],
              colWidths=[3.5 * cm] * 5)
    k.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SOFT), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 1), (-1, 1), 7)]))
    story += [k, Spacer(1, 8), P("Analyse", "h2")] + [P(t) for t in p["texte"]]
    h = p["historique"].dropna()
    if len(h):
        reg = st.countries[st.countries.region == an["pays"]["region"]]["iso3"]
        rh = st.pillars_hist[(st.pillars_hist.pillar == pid) & st.pillars_hist.iso3.isin(reg)].groupby("year")["score"].mean()
        story.append(IMG(im.series_chart(h, f"Score du pilier {p['nom']}, {int(h.index.min())}–{int(h.index.max())}", "score /100",
                                         ref=rh, ref_label="moyenne régionale", color=im.score_color(p["score"])), 16.5))
    story += [P("Indicateurs (dernière valeur disponible)", "h2"), _indicator_table(p), PageBreak(),
              P("Séries longues (jusqu'à 50 ans) — pays vs médiane régionale", "h2")]
    reg = st.countries[st.countries.region == an["pays"]["region"]]["iso3"]
    for _, r in p["indicateurs"].iterrows():
        d = ind_def(r.code)
        s = per_capita_series(st, iso, r.code)
        if len(s) < 3:
            continue
        sub = st.long[(st.long.indicator == r.code) & st.long.iso3.isin(reg)]
        refm = sub.groupby("year")["value"].median() if not d.get("per_capita") else None
        story.append(IMG(im.series_chart(s, f"{d['label']} ({d['unit']})", d["unit"], ref=refm), 15.5))
        story.append(P(f"Source : {r.source} — code {r.code} — {r.url}", "small"))
    pr = st.proj_ind[(st.proj_ind.iso3 == iso) & (st.proj_ind.pillar == pid) & (st.proj_ind.horizon == 2050)]
    if len(pr):
        piv = pr.pivot_table(index="indicator", columns="scenario", values="score")
        rows = [["Indicateur", "Score actuel", "2050 pessimiste", "2050 central", "2050 optimiste"]]
        cur = p["indicateurs"].set_index("code")["score"]
        for code, r in piv.iterrows():
            rows.append([ind_def(code)["label"], fnum(cur.get(code), 0), fnum(r.get("pessimiste"), 0), fnum(r.get("central"), 0), fnum(r.get("optimiste"), 0)])
        story += [P("Projections 2050 par indicateur (scores 0-100)", "h2"), _table(rows, [7, 2.5, 2.7, 2.5, 2.7])]
    story += [P("Sources", "h2"), _sources(an, pid), Spacer(1, 6), P("Pour aller plus loin — analyses d'experts", "h3")] + _experts(an, [pid]) + _methodo()
    return D.build(story)


# --------------------------------------------------------------------------- rapport élections
def elections_report(st, an: dict, calendar: pd.DataFrame | None, polls: dict | None, news: dict | None) -> bytes:
    iso, name = an["iso3"], an["nom"]
    D = _Doc(f"Élections & risque politique — {name}", "")
    story = [P(f"Élections & risque politique — {name}", "h1"), _kpis(an["master"]), Spacer(1, 8),
             P("Diagnostic", "h2")] + [P(t) for t in an["politique"]["texte"]]
    if an["politique"]["facteurs"]:
        rows = [["Facteur", "Contribution estimée"]] + [[k, f"{'+' if v >= 0 else ''}{fnum(v * 100, 1)} pt(s)"] for k, v in an["politique"]["facteurs"]]
        story += [P("Facteurs qui pèsent le plus dans la prévision du modèle", "h3"), _table(rows, [8, 6])]
    story += [P("Déterminants structurels du vote", "h2")]
    for pid in ("impunite", "corruption", "economie", "social", "education", "securite"):
        p = an["piliers"][pid]
        story += [P(f"{p['nom']} — {fnum(p['score'], 0)}/100", "h3")] + [P(t) for t in p["texte"][:4]]
    for code, lab in (("v2x_polyarchy", "Démocratie électorale (V-Dem)"), ("v2cacamps", "Polarisation politique (V-Dem)"),
                      ("fp_cpi_totl_zg", "Inflation (%)")):
        s = st.series(iso, code)
        if len(s) > 5:
            story.append(IMG(im.series_chart(s, lab, ""), 15))
    story += [PageBreak(), P("Calendrier et sondages", "h2")]
    if calendar is not None and len(calendar):
        rows = [["Date", "Scrutin", "Statut", "Source"]] + [[d.strftime("%d/%m/%Y"), e, s, u] for d, e, s, u in
                                                            zip(calendar["date"], calendar["election"], calendar["statut"], calendar["source"])]
        story.append(_table(rows, [2.4, 6.5, 4.4, 4.3]))
    else:
        story.append(P("Aucun scrutin national enregistré dans le calendrier pour ce pays (modifiable dans l'onglet Élections)."))
    if polls:
        t = polls["table"]
        rows = [["Candidat / parti", "Moyenne pondérée", "Intervalle 80 %", "Probabilité d'arriver en tête"]]
        rows += [[i, f"{r.moyenne_ponderee:.1f} %", f"{r.intervalle_80_bas:.0f}–{r.intervalle_80_haut:.0f} %", f"{r.proba_en_tete * 100:.0f} %"] for i, r in t.iterrows()]
        story += [P("Agrégation des sondages", "h3"), _table(rows, [6, 3.5, 3.5, 4.6]),
                  P("Méthode : moyenne pondérée par la récence (demi-vie 14 jours) et la taille d'échantillon, puis 20 000 simulations "
                    "avec une erreur de sondage de 3,5 points et une erreur commune corrélée.", "small")]
    else:
        story.append(P("Aucun sondage saisi : ajoutez des sondages (institut, date, échantillon, intentions) dans l'onglet Élections "
                       "pour obtenir des probabilités par candidat."))
    if news and news.get("ok") and news.get("n"):
        t = news["table"]
        decl = t[t["declaration"]].head(10)
        story += [P("Déclarations politiques récentes (presse)", "h3")]
        for _, r in decl.iterrows():
            story.append(Paragraph(f"• {escape(str(r['titre']))}", S["small"]))
    story += [P("Sources", "h2"), P("V-Dem Institute (v16), Banque mondiale (WDI), Gapminder (EIU, IDEA, Transparency International, "
                                   "World Happiness Report), calendrier électoral saisi/Wikipédia, sondages saisis par l'utilisateur, "
                                   "Google News.", "small")] + _methodo()
    return D.build(story)
