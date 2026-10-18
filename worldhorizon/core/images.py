"""Génération d'images (PNG) : affiche pays, radar, cartes, séries longues, projections.
Utilisé par l'application (téléchargement) et par les rapports PDF. Aucune API externe requise."""
from __future__ import annotations

import io
import json
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from .catalog import AGENTS, COLOR_THRESHOLDS  # noqa: E402
from .paths import REF  # noqa: E402

PALETTE = {"vert": "#1fa37a", "orange": "#e8962e", "rouge": "#d64545", "gris": "#8a93a3"}
INK = "#0f1b2d"
MUTED = "#5b6b82"
ACCENT = "#2f6fde"
BG_DARK = "#0b1220"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c9d2df", "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlecolor": INK})


SHORT = {"sante": "Santé", "education": "Éducation", "economie": "Économie", "finance": "Finance",
         "business": "Business", "environnement": "Environnement", "agriculture": "Agriculture",
         "gouvernance": "Gouvernance", "impunite": "Justice/Impunité", "corruption": "Corruption",
         "securite": "Sécurité", "social": "Social", "innovation": "Innovation", "tourisme": "Tourisme"}


def score_color(s) -> str:
    if s is None or not np.isfinite(s):
        return PALETTE["gris"]
    return PALETTE["vert"] if s >= COLOR_THRESHOLDS["vert"] else PALETTE["orange"] if s >= COLOR_THRESHOLDS["orange"] else PALETTE["rouge"]


def to_png(fig, dpi=170) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


@lru_cache(maxsize=1)
def world_geo() -> dict:
    p = REF / "world.geo.json"
    return json.loads(p.read_text()) if p.exists() else {"features": []}


def _polys(feature):
    g = feature["geometry"]
    if g["type"] == "Polygon":
        return [np.array(g["coordinates"][0])]
    return [np.array(poly[0]) for poly in g["coordinates"]]


def radar(scores: dict, title: str = "", color: str = ACCENT, compare: dict | None = None, dark=False) -> bytes:
    labels = [SHORT[a["id"]] for a in AGENTS]
    vals = [scores.get(a["id"], np.nan) for a in AGENTS]
    v = np.nan_to_num(np.array(vals, dtype=float), nan=0)
    ang = np.linspace(0, 2 * np.pi, len(v), endpoint=False)
    fig = plt.figure(figsize=(5.4, 5.4), facecolor=BG_DARK if dark else "white")
    ax = fig.add_subplot(111, polar=True, facecolor=BG_DARK if dark else "white")
    a2 = np.r_[ang, ang[:1]]
    if compare:
        cv = np.nan_to_num(np.array([compare.get(a["id"], np.nan) for a in AGENTS], dtype=float), nan=0)
        ax.plot(a2, np.r_[cv, cv[:1]], color="#9aa7bb", lw=1.2, ls="--", label="moyenne mondiale")
    ax.fill(a2, np.r_[v, v[:1]], color=color, alpha=0.25)
    ax.plot(a2, np.r_[v, v[:1]], color=color, lw=2.2)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels, fontsize=8, color="#d7e0ee" if dark else INK)
    ax.set_ylim(0, 100); ax.set_yticks([25, 50, 75]); ax.set_yticklabels(["25", "50", "75"], fontsize=7, color=MUTED)
    ax.grid(color="#2a3a55" if dark else "#dfe5ee")
    ax.spines["polar"].set_color("#2a3a55" if dark else "#dfe5ee")
    if title:
        ax.set_title(title, fontsize=11, pad=18, color="white" if dark else INK)
    if compare:
        ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.08), fontsize=7, frameon=False)
    return to_png(fig)


def pillar_bars(scores: dict, ranks: dict | None = None) -> bytes:
    items = [(a["nom"], scores.get(a["id"], np.nan)) for a in AGENTS]
    items = sorted(items, key=lambda t: -np.nan_to_num(t[1], nan=-1))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(items))[::-1]
    for yi, (lab, s) in zip(y, items):
        ax.barh(yi, 100, color="#eef2f7", height=0.62)
        if np.isfinite(s):
            ax.barh(yi, s, color=score_color(s), height=0.62)
            ax.text(s + 1.5, yi, f"{s:.0f}", va="center", fontsize=8, color=INK, fontweight="bold")
        else:
            ax.text(2, yi, "n.d.", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([t[0] for t in items], fontsize=8.5)
    ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100]); ax.tick_params(axis="x", labelsize=7)
    ax.axvline(COLOR_THRESHOLDS["orange"], color="#c9d2df", lw=0.8, ls=":")
    ax.axvline(COLOR_THRESHOLDS["vert"], color="#c9d2df", lw=0.8, ls=":")
    ax.spines["left"].set_visible(False)
    return to_png(fig)


def series_chart(s: pd.Series, title: str, unit: str, ref: pd.Series | None = None, ref_label="médiane régionale",
                 color=ACCENT, proj: dict | None = None) -> bytes:
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    if ref is not None and len(ref):
        ax.plot(ref.index, ref.values, color="#9aa7bb", lw=1.2, ls="--", label=ref_label)
    ax.plot(s.index, s.values, color=color, lw=2.2, label="pays")
    if len(s):
        ax.scatter([s.index[-1]], [s.values[-1]], color=color, s=18, zorder=5)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_ylabel(unit, fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="#eef2f7")
    ax.legend(fontsize=7, frameon=False, loc="best")
    return to_png(fig, dpi=150)


def history_projection(hist: pd.Series, proj: pd.DataFrame | None, title="Score de situation : 1975 → 2050") -> bytes:
    fig, ax = plt.subplots(figsize=(7, 2.9))
    if len(hist):
        ax.plot(hist.index, hist.values, color=INK, lw=2.2, label="historique")
    if proj is not None and len(proj) and len(hist):
        x0, y0 = hist.index[-1], hist.values[-1]
        piv = proj.set_index(["horizon", "scenario"])["score_actuel"]
        xs = [x0] + sorted(proj["horizon"].unique())
        for scen, col in (("pessimiste", PALETTE["rouge"]), ("central", ACCENT), ("optimiste", PALETTE["vert"])):
            ys = [y0] + [piv.get((h, scen), np.nan) for h in xs[1:]]
            ax.plot(xs, ys, color=col, lw=1.8, ls="--" if scen != "central" else "-", label=f"scénario {scen}")
        lo = [y0] + [piv.get((h, "pessimiste"), np.nan) for h in xs[1:]]
        hi = [y0] + [piv.get((h, "optimiste"), np.nan) for h in xs[1:]]
        ax.fill_between(xs, lo, hi, color=ACCENT, alpha=0.08)
    ax.axhspan(COLOR_THRESHOLDS["vert"], 100, color=PALETTE["vert"], alpha=0.05)
    ax.axhspan(0, COLOR_THRESHOLDS["orange"], color=PALETTE["rouge"], alpha=0.05)
    ax.set_ylim(0, 100); ax.set_title(title, fontsize=10, loc="left"); ax.tick_params(labelsize=7)
    ax.grid(axis="y", color="#eef2f7"); ax.legend(fontsize=7, frameon=False, ncol=4, loc="lower left")
    return to_png(fig)


def population_chart(pop: pd.Series, name: str) -> bytes:
    fig, ax = plt.subplots(figsize=(7, 2.6))
    hist = pop[pop.index <= 2025]; fut = pop[pop.index >= 2025]
    ax.plot(hist.index, hist.values / 1e6, color=INK, lw=2.2, label="observé")
    ax.plot(fut.index, fut.values / 1e6, color=ACCENT, lw=2.2, ls="--", label="projection ONU")
    ax.set_title(f"Population de {name}, 1970 → 2100 (millions)", fontsize=10, loc="left")
    ax.tick_params(labelsize=7); ax.grid(axis="y", color="#eef2f7"); ax.legend(fontsize=7, frameon=False)
    return to_png(fig)


def future_components(comps: dict) -> bytes:
    items = [(k, v) for k, v in comps.items() if v is not None and np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for i, (k, v) in enumerate(items):
        ax.barh(i, 100, color="#eef2f7", height=0.55)
        ax.barh(i, v, color=score_color(v), height=0.55)
        ax.text(v + 1.5, i, f"{v:.0f}", va="center", fontsize=8, fontweight="bold", color=INK)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([k for k, _ in items], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0, 108); ax.tick_params(axis="x", labelsize=7); ax.spines["left"].set_visible(False)
    ax.set_title("Composantes de l'indice d'avenir (horizon 2100)", fontsize=10, loc="left")
    return to_png(fig)


def world_map(scores: pd.Series, highlight: str | None = None, title: str = "", dark=False) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor=BG_DARK if dark else "white")
    ax.set_facecolor(BG_DARK if dark else "white")
    patches, cols, edges = [], [], []
    for f in world_geo()["features"]:
        iso = f["id"]
        for ring in _polys(f):
            patches.append(Polygon(ring, closed=True))
            cols.append(score_color(scores.get(iso, np.nan)))
            edges.append("#ffffff" if iso != highlight else "#111111")
    pc = PatchCollection(patches, facecolor=cols, edgecolor=edges, linewidths=0.25)
    ax.add_collection(pc)
    if highlight:
        for f in world_geo()["features"]:
            if f["id"] == highlight:
                for ring in _polys(f):
                    ax.add_patch(Polygon(ring, closed=True, fill=False, edgecolor="#ffd84d" if dark else "#111", lw=1.6))
    ax.set_xlim(-170, 190); ax.set_ylim(-58, 85); ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, color="white" if dark else INK, loc="left")
    return to_png(fig, dpi=150)


def country_poster(iso: str, name: str, score: float, actuel: float, avenir: float, pillars: dict,
                   subtitle: str = "") -> bytes:
    """Image de couverture : silhouette du pays aux couleurs de son score, chiffres clés, radar."""
    col = score_color(score)
    fig = plt.figure(figsize=(12, 6.3), facecolor=BG_DARK)
    # dégradé de fond
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    grad = np.linspace(0, 1, 256)[None, :]
    bg.imshow(np.vstack([grad] * 2), cmap=matplotlib.colors.LinearSegmentedColormap.from_list("g", [BG_DARK, "#15233d"]),
              aspect="auto", extent=[0, 1, 0, 1])
    # grille de méridiens stylisée
    for x in np.linspace(0, 1, 25):
        bg.axvline(x, color="#ffffff", alpha=0.03, lw=0.6)
    for y in np.linspace(0, 1, 13):
        bg.axhline(y, color="#ffffff", alpha=0.03, lw=0.6)
    feat = next((f for f in world_geo()["features"] if f["id"] == iso), None)
    ax = fig.add_axes([0.03, 0.08, 0.45, 0.84]); ax.axis("off"); ax.set_facecolor("none")
    if feat:
        rings = _polys(feat)
        allp = np.vstack(rings)
        # ignorer les territoires très éloignés (ex. outre-mer) pour cadrer la silhouette
        main = max(rings, key=lambda r: len(r))
        cx, cy = main[:, 0].mean(), main[:, 1].mean()
        keep = [r for r in rings if np.hypot(r[:, 0].mean() - cx, r[:, 1].mean() - cy) < 25] or rings
        for r in keep:
            for lw, a in ((14, 0.05), (8, 0.08), (3, 0.18)):
                ax.add_patch(Polygon(r, closed=True, fill=False, edgecolor=col, lw=lw, alpha=a))
            ax.add_patch(Polygon(r, closed=True, facecolor=col, edgecolor="white", lw=0.8, alpha=0.92))
        pts = np.vstack(keep)
        pad = 0.08 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1)
        ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad); ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
        ax.set_aspect("equal")
    else:
        ax.add_patch(matplotlib.patches.Circle((0.5, 0.5), 0.32, color=col, alpha=0.9, transform=ax.transAxes))
        ax.text(0.5, 0.5, iso, ha="center", va="center", fontsize=40, color="white", fontweight="bold", transform=ax.transAxes)
    t = fig.add_axes([0.5, 0.05, 0.48, 0.9]); t.axis("off")
    t.text(0, 0.93, "WORLDHORIZON · ANALYSE PAYS", color="#7f93b5", fontsize=10, fontweight="bold")
    t.text(0, 0.80, name, color="white", fontsize=34 if len(name) < 16 else 24, fontweight="bold")
    if subtitle:
        t.text(0, 0.72, subtitle, color="#a9b8d0", fontsize=11)
    t.text(0, 0.47, f"{score:.0f}" if np.isfinite(score) else "n.d.", color=col, fontsize=78, fontweight="bold")
    t.text(0.30, 0.52, "/100\nscore global", color="#c8d3e6", fontsize=12)
    t.text(0, 0.34, f"Situation actuelle  {actuel:.0f}" if np.isfinite(actuel) else "Situation actuelle  n.d.", color="white", fontsize=12)
    t.text(0, 0.27, f"Indice d'avenir 2100  {avenir:.0f}" if np.isfinite(avenir) else "Indice d'avenir 2100  n.d.", color="white", fontsize=12)
    # mini barres des piliers
    bx = fig.add_axes([0.5, 0.06, 0.46, 0.17]); bx.axis("off"); bx.set_facecolor("none")
    vals = [pillars.get(a["id"], np.nan) for a in AGENTS]
    for i, (a, v) in enumerate(zip(AGENTS, vals)):
        bx.bar(i, 100, color="#1d2b45", width=0.7)
        if np.isfinite(v):
            bx.bar(i, v, color=score_color(v), width=0.7)
        bx.text(i, -10, SHORT[a["id"]][:4], ha="center", va="top", fontsize=7.5, color="#a9b8d0", rotation=0)
    bx.set_ylim(-28, 100); bx.set_xlim(-0.6, len(AGENTS) - 0.4)
    return to_png(fig, dpi=130)
