"""Moteur de Machine Learning WorldHorizon.

1. Trajectoires (2035 / 2050) : tendance linéaire sur 25 ans, amortie (demi-vie 15 ans),
   avec trois scénarios (pessimiste / central / optimiste) tirés de la dispersion des résidus.
2. Indice d'avenir (horizon 2100) : démographie (projections ONU), capital humain,
   institutions (niveau + tendance 20 ans), climat & ressources, dynamique économique.
3. Modèle de risque politique : classifieur gradient boosting entraîné sur le panel
   1980-aujourd'hui (V-Dem + Banque mondiale) pour prédire un choc politique majeur dans les 3 ans
   (recul marqué de la démocratie électorale ou de la qualité des élections). Validation temporelle :
   entraînement ≤ 2008, test 2009 → année-4.
4. Pays comparables : partitionnement k-means sur les 14 piliers + plus proches voisins.

À chaque mise à jour, les modèles sont réentraînés sur les CSV enrichis et les métriques
sont ajoutées à data/ml/metrics_log.csv : la plateforme « apprend » au fil des collectes.
"""
from __future__ import annotations

import datetime as dt
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .catalog import AGENTS, FUTURE_WEIGHTS
from .paths import ML_DIR, MODELS
from .scoring import CURRENT_YEAR, YEARS, _wide, aggregate, transform

warnings.filterwarnings("ignore")
HORIZONS = [2035, 2050]
DAMPING_HALF_LIFE = 15  # années : la pente tendancielle s'amortit de moitié tous les 15 ans


# --------------------------------------------------------------------------- trajectoires
def _vector_trend(t: pd.DataFrame):
    """Pente OLS, niveau récent et écart-type des résidus, ligne par ligne (NaN tolérés)."""
    cols = [c for c in t.columns if c >= CURRENT_YEAR - 25]
    Y = t[cols].to_numpy(dtype=float)
    X = np.array(cols, dtype=float)[None, :].repeat(len(Y), 0)
    M = ~np.isnan(Y)
    n = M.sum(1)
    Xm = np.where(M, X, 0).sum(1) / np.maximum(n, 1)
    Ym = np.where(M, Y, 0).sum(1) / np.maximum(n, 1)
    dx = np.where(M, X - Xm[:, None], 0); dy = np.where(M, Y - Ym[:, None], 0)
    slope = (dx * dy).sum(1) / np.maximum((dx ** 2).sum(1), 1e-9)
    resid = np.where(M, dy - slope[:, None] * dx, np.nan)
    sd = np.nanstd(resid, axis=1)
    last_year = np.where(M, X, -np.inf).max(1)
    # niveau de départ : moyenne des 3 dernières observations
    base = np.array([row[m][-3:].mean() if m.any() else np.nan for row, m in zip(Y, M)])
    ok = n >= 8
    return slope, base, sd, last_year, ok


def project_scores(scorer) -> pd.DataFrame:
    """Projette chaque indicateur noté (en échelle transformée) puis recalcule
    les scores de pilier et le score actuel à 2035 et 2050 pour 3 scénarios."""
    gp = {g["indicator"]: g for g in scorer.goalposts}
    k = np.log(2) / DAMPING_HALF_LIFE
    frames = []
    done = set()
    for agent in AGENTS:
        for ind in agent["indicateurs"]:
            code = ind["code"]
            if code not in gp or ind["src"] == "pisa" or ind["weight"] == 0:
                continue
            g = gp[code]
            t = transform(scorer.raw[code], ind)
            slope, base, sd, ly, ok = _vector_trend(t)
            for h in HORIZONS:
                n = h - ly
                eff = slope * (1 - np.exp(-k * n)) / k
                spread = sd * np.sqrt(np.maximum(n, 1) / 5) + np.abs(slope) * n * 0.25
                for scen, v in (("bas", base + eff - spread), ("central", base + eff), ("haut", base + eff + spread)):
                    sc = np.clip((v - g["lo"]) / (g["hi"] - g["lo"]), 0, 1) * 100
                    if ind["direction"] < 0:
                        sc = 100 - sc
                    frames.append(pd.DataFrame({"iso3": t.index, "pillar": agent["id"], "indicator": code,
                                                "horizon": h, "scenario": scen, "score": sc,
                                                "weight": ind["weight"]})[ok])
    df = pd.concat(frames, ignore_index=True)
    piv = df.pivot_table(index=["iso3", "pillar", "indicator", "horizon", "weight"], columns="scenario", values="score").reset_index()
    # « pessimiste » = la borne défavorable, quel que soit le sens de l'indicateur
    piv["pessimiste"] = piv[["bas", "haut"]].min(axis=1)
    piv["optimiste"] = piv[["bas", "haut"]].max(axis=1)
    long = piv.melt(id_vars=["iso3", "pillar", "indicator", "horizon", "weight"], value_vars=["pessimiste", "central", "optimiste"],
                    var_name="scenario", value_name="score")
    long["ws"] = long["score"] * long["weight"]
    pil = long.groupby(["iso3", "horizon", "scenario", "pillar"]).agg(ws=("ws", "sum"), w=("weight", "sum")).reset_index()
    pil["score"] = pil["ws"] / pil["w"]
    out = []
    for (h, s), g in pil.groupby(["horizon", "scenario"]):
        wide = g.pivot(index="iso3", columns="pillar", values="score")
        for a in AGENTS:
            if a["id"] not in wide:
                wide[a["id"]] = np.nan
        o = wide.copy(); o["score_actuel"] = aggregate(wide); o["horizon"] = h; o["scenario"] = s
        out.append(o.reset_index())
    res = pd.concat(out, ignore_index=True)
    res.round(2).to_csv(ML_DIR / "projections_scores.csv", index=False)
    long.round(2).to_csv(ML_DIR / "projections_indicators.csv", index=False)
    return res


# --------------------------------------------------------------------------- indice d'avenir
def _band_score(x, good_lo, good_hi, bad_lo, bad_hi):
    """100 dans [good_lo, good_hi], décroît linéairement jusqu'à 0 en bad_lo / bad_hi."""
    x = pd.Series(x, dtype=float)
    s = pd.Series(100.0, index=x.index)
    s = s.where(x >= good_lo, 100 * (x - bad_lo) / (good_lo - bad_lo))
    s = s.where(x <= good_hi, 100 * (bad_hi - x) / (bad_hi - good_hi))
    return s.clip(0, 100).where(x.notna())


def _pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True) * 100


def future_index(long: pd.DataFrame, pillars: pd.DataFrame, proj: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    iso = pillars.index
    pop = _wide(long, "pop").reindex(iso)
    lex = _wide(long, "lex").reindex(iso)
    pop_full = long[long.indicator == "pop"].pivot_table(index="iso3", columns="year", values="value").reindex(iso)
    lex_full = long[long.indicator == "lex"].pivot_table(index="iso3", columns="year", values="value").reindex(iso)
    tfr = _wide(long, "sp_dyn_tfrt_in").reindex(iso).ffill(axis=1, limit=6)[CURRENT_YEAR]
    p_now = pop_full.get(2025, pop[CURRENT_YEAR] if CURRENT_YEAR in pop else np.nan)
    p2050 = pop_full.get(2050); p2100 = pop_full.get(2100)
    l2100 = lex_full.get(2100)

    d = pd.DataFrame(index=iso)
    d["pop_2025"] = p_now; d["pop_2050"] = p2050; d["pop_2100"] = p2100
    d["var_pop_2100_pct"] = (p2100 / p_now - 1) * 100
    d["lex_2100"] = l2100
    d["fecondite"] = tfr
    demo = (0.45 * _band_score(d["var_pop_2100_pct"], -10, 60, -55, 220).fillna(50)
            + 0.30 * _band_score(tfr, 1.8, 2.4, 0.8, 5.5).fillna(50)
            + 0.25 * _band_score(l2100, 85, 120, 65, 130).fillna(50))
    d["demographie"] = demo.where(d["pop_2025"].notna() | tfr.notna())

    # tendance 20 ans des piliers (points de score par décennie)
    h = history.pivot_table(index=["iso3", "pillar"], columns="year", values="score")
    def trend(pillar):
        sub = h.xs(pillar, level="pillar") if pillar in h.index.get_level_values("pillar") else pd.DataFrame()
        if sub.empty:
            return pd.Series(np.nan, index=iso)
        a = sub.get(CURRENT_YEAR - 1); b = sub.get(CURRENT_YEAR - 21)
        return ((a - b) / 2).reindex(iso)
    d["tendance_institutions"] = pd.concat([trend("gouvernance"), trend("impunite"), trend("corruption")], axis=1).mean(axis=1)
    d["tendance_sante"] = trend("sante")
    d["tendance_economie"] = trend("economie")
    d["tendance_environnement"] = trend("environnement")

    edu = pillars["education"]; sante = pillars["sante"]
    d["capital_humain"] = (0.6 * edu.fillna(sante) + 0.25 * sante + 0.15 * (50 + d["tendance_sante"].fillna(0) * 5).clip(0, 100))
    inst_level = pillars[["gouvernance", "impunite", "corruption"]].mean(axis=1)
    d["institutions"] = (0.75 * inst_level + 0.25 * (50 + d["tendance_institutions"].fillna(0) * 6).clip(0, 100))
    d["climat_ressources"] = (0.45 * pillars["environnement"] + 0.35 * pillars["agriculture"]
                              + 0.20 * (50 + d["tendance_environnement"].fillna(0) * 6).clip(0, 100))
    p50 = proj[(proj.horizon == 2050) & (proj.scenario == "central")].set_index("iso3")["score_actuel"].reindex(iso)
    gdp = _wide(long, "ny_gdp_pcap_pp_kd").reindex(iso)
    g_now = gdp.ffill(axis=1, limit=4)[CURRENT_YEAR]; g_past = gdp.ffill(axis=1, limit=4)[CURRENT_YEAR - 20]
    cagr = ((g_now / g_past) ** (1 / 20) - 1) * 100
    d["croissance_tendancielle_pct"] = cagr
    d["dynamique_eco"] = (0.35 * _band_score(cagr, 2.5, 8, -2, 12).fillna(40) + 0.35 * pillars["innovation"].fillna(30)
                          + 0.30 * p50.fillna(pillars[["economie", "innovation"]].mean(axis=1)))
    comp = list(FUTURE_WEIGHTS)
    w = pd.Series(FUTURE_WEIGHTS)
    num = (d[comp] * w).sum(axis=1, min_count=3)
    den = d[comp].notna().mul(w).sum(axis=1)
    d["indice_avenir"] = (num / den).where(d[comp].notna().sum(axis=1) >= 4)
    d["score_2050_pessimiste"] = proj[(proj.horizon == 2050) & (proj.scenario == "pessimiste")].set_index("iso3")["score_actuel"].reindex(iso)
    d["score_2050_central"] = p50
    d["score_2050_optimiste"] = proj[(proj.horizon == 2050) & (proj.scenario == "optimiste")].set_index("iso3")["score_actuel"].reindex(iso)
    d.index.name = "iso3"
    d.round(3).to_csv(ML_DIR / "future_index.csv")
    return d


# --------------------------------------------------------------------------- risque politique
FEATURES = {
    "inflation": ("fp_cpi_totl_zg", lambda v: np.sign(v) * np.log1p(np.abs(v))),
    "croissance": ("ny_gdp_mktp_kd_zg", None),
    "pib_hab_log": ("ny_gdp_pcap_pp_kd", np.log10),
    "chomage": ("sl_uem_totl_zs", None),
    "chomage_jeunes": ("sl_uem_1524_zs", None),
    "homicides_log": ("vc_ihr_psrc_p5", np.log1p),
    "morts_combat_log": ("vc_btl_deth", np.log1p),
    "corruption_politique": ("v2x_corr", None),
    "corruption_executif": ("v2x_execorr", None),
    "etat_de_droit": ("v2x_rule", None),
    "controle_judiciaire": ("v2x_jucon", None),
    "democratie_electorale": ("v2x_polyarchy", None),
    "elections_libres": ("v2xel_frefair", None),
    "libertes_civiles": ("v2x_civlib", None),
    "polarisation": ("v2cacamps", None),
    "polarisation_societe": ("v2smpolsoc", None),
    "croissance_pop": ("sp_pop_grow", None),
    "pop_65": ("sp_pop_65up_to_zs", None),
    "urbanisation": ("sp_urb_totl_in_zs", None),
    "gini": ("si_pov_gini", None),
}
FEATURE_LABELS = {
    "inflation": "Inflation", "croissance": "Croissance du PIB", "pib_hab_log": "Niveau de vie",
    "chomage": "Chômage", "chomage_jeunes": "Chômage des jeunes", "homicides_log": "Homicides",
    "morts_combat_log": "Conflit armé",
    "corruption_politique": "Corruption politique", "corruption_executif": "Corruption de l'exécutif",
    "etat_de_droit": "État de droit", "controle_judiciaire": "Contrôle judiciaire de l'exécutif",
    "democratie_electorale": "Niveau de démocratie électorale", "elections_libres": "Élections libres et équitables",
    "libertes_civiles": "Libertés civiles", "polarisation": "Polarisation politique",
    "polarisation_societe": "Polarisation de la société",
    "croissance_pop": "Croissance démographique", "pop_65": "Vieillissement", "urbanisation": "Urbanisation",
    "gini": "Inégalités (Gini)", "d3_democratie_electorale": "Tendance 3 ans de la démocratie",
    "d3_corruption_politique": "Tendance 3 ans de la corruption", "d3_inflation": "Accélération de l'inflation",
    "d3_polarisation": "Tendance 3 ans de la polarisation",
}


def _panel(long: pd.DataFrame, iso: list[str]) -> pd.DataFrame:
    cols = {}
    for name, (code, f) in FEATURES.items():
        w = _wide(long, code).reindex(iso)
        w = w.ffill(axis=1, limit=4 if code not in ("si_pov_gini",) else 10)
        if f is not None:
            w = f(w.clip(lower=-0.99) if f is np.log1p else w)
        cols[name] = w.stack(future_stack=True)
    p = pd.DataFrame(cols)
    p.index.names = ["iso3", "year"]
    p = p.reset_index()
    p = p.sort_values(["iso3", "year"])
    for c in ("democratie_electorale", "corruption_politique", "inflation", "polarisation"):
        p["d3_" + c] = p.groupby("iso3")[c].diff(3)
    # cible : choc politique majeur dans les 3 années suivantes = recul marqué de la démocratie
    # électorale (V-Dem polyarchy -0,05) ou de la qualité des élections (-0,10)
    rg = _wide(long, "v2x_polyarchy").reindex(iso)
    ce = _wide(long, "v2xel_frefair").reindex(iso)
    fut_min_rg = pd.concat([rg.shift(-k, axis=1) for k in (1, 2, 3)]).groupby(level=0).min()
    fut_min_ce = pd.concat([ce.shift(-k, axis=1) for k in (1, 2, 3)]).groupby(level=0).min()
    shock = ((rg - fut_min_rg) >= 0.05) | ((ce - fut_min_ce) >= 0.10)
    label = shock.where(fut_min_rg.notna() & rg.notna()).stack(future_stack=True).rename("choc")
    label.index.names = ["iso3", "year"]
    p = p.merge(label.reset_index(), on=["iso3", "year"], how="left")
    return p


def train_political_model(long: pd.DataFrame, iso: list[str]) -> tuple[pd.DataFrame, dict]:
    p = _panel(long, iso)
    feats = [c for c in p.columns if c not in ("iso3", "year", "choc")]
    lab = p.dropna(subset=["choc"])
    lab = lab[(lab.year >= 1980) & (lab[feats].notna().sum(axis=1) >= 8)]
    y = lab["choc"].astype(int)
    train = lab.year <= 2008
    feats = [f for f in feats if lab.loc[train, f].nunique() > 5]
    test = (lab.year > 2008) & (lab.year <= CURRENT_YEAR - 4)
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=350,
                                         l2_regularization=1.0, class_weight="balanced", random_state=7)
    clf.fit(lab.loc[train, feats], y[train])
    proba_test = clf.predict_proba(lab.loc[test, feats])[:, 1]
    auc = roc_auc_score(y[test], proba_test) if y[test].nunique() > 1 else np.nan
    imp = permutation_importance(clf, lab.loc[test, feats], y[test], scoring="roc_auc", n_repeats=5, random_state=7)
    importances = pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False)
    # modèle final sur tout l'historique, probabilités calibrées par la fréquence de base
    final = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=350,
                                           l2_regularization=1.0, random_state=7)
    final.fit(lab[feats], y)
    brier = brier_score_loss(y[test], clf.predict_proba(lab.loc[test, feats])[:, 1])
    joblib.dump({"model": final, "features": feats}, MODELS / "political_risk.joblib")

    latest = p[p.year == CURRENT_YEAR].set_index("iso3")
    # repli sur l'année précédente si nécessaire
    prev = p[p.year == CURRENT_YEAR - 1].set_index("iso3")
    X = latest[feats].combine_first(prev[feats])
    ok = X.notna().sum(axis=1) >= 8
    risk = pd.Series(np.nan, index=X.index)
    risk[ok] = final.predict_proba(X[ok])[:, 1]
    # contributions locales approximées : écart à la médiane × importance
    med = lab[feats].median()
    contrib = {}
    for iso3 in X.index[ok]:
        x = X.loc[iso3]
        delta = []
        for f in importances.index[:10]:
            if pd.isna(x[f]):
                continue
            z = x.copy(); z[f] = med[f]
            pz = final.predict_proba(z.to_frame().T[feats])[:, 1][0]
            delta.append((f, risk[iso3] - pz))
        delta.sort(key=lambda t: -abs(t[1]))
        contrib[iso3] = ";".join(f"{f}:{v:+.3f}" for f, v in delta[:5])
    out = pd.DataFrame({"proba_choc_3ans": risk})
    out["facteurs"] = pd.Series(contrib)
    out.index.name = "iso3"
    metrics = dict(model="HistGradientBoosting", target="recul démocratique / choc politique à 3 ans",
                   n_train=int(train.sum()), n_test=int(test.sum()), base_rate=float(y.mean()),
                   auc_test=float(auc), brier_test=float(brier))
    importances.rename("importance").to_csv(ML_DIR / "political_feature_importance.csv")
    return out, metrics


# --------------------------------------------------------------------------- pays comparables
def peers(pillars: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    P = pillars.dropna(thresh=9)
    X = P.fillna(P.median())
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=7).fit(Xs)
    D = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)
    out = []
    for i, iso in enumerate(X.index):
        order = [X.index[j] for j in np.argsort(D[i]) if j != i][:5]
        out.append((iso, int(km.labels_[i]), ";".join(order)))
    df = pd.DataFrame(out, columns=["iso3", "groupe", "pays_proches"]).set_index("iso3")
    joblib.dump(km, MODELS / "peers_kmeans.joblib")
    df.to_csv(ML_DIR / "peers.csv")
    return df


def run_ml(long: pd.DataFrame, scores: dict, log=print) -> dict:
    t0 = dt.datetime.utcnow()
    log("Machine learning : projections 2035/2050 des indicateurs…")
    proj = project_scores(scores["scorer"])
    log("Machine learning : indice d'avenir (horizon 2100)…")
    fut = future_index(long, scores["pillars"], proj, scores["history"])
    log("Machine learning : entraînement du modèle de risque politique…")
    risk, metrics = train_political_model(long, list(scores["pillars"].index))
    risk.round(4).to_csv(ML_DIR / "political_risk.csv")
    log(f"   AUC hors échantillon (test 2009 → année-4) = {metrics['auc_test']:.3f}")
    log("Machine learning : pays comparables (k-means)…")
    pe = peers(scores["pillars"])
    metrics.update(run_at=t0.strftime("%Y-%m-%dT%H:%M:%SZ"), n_countries_scored=int(scores["current"].notna().sum()),
                   n_projections=int(len(proj)))
    log_path = ML_DIR / "metrics_log.csv"
    pd.DataFrame([metrics]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    return dict(projections=proj, future=fut, risk=risk, metrics=metrics, peers=pe)
