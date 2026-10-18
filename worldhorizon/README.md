# WorldHorizon

Plateforme d'analyse holistique de **251 pays et territoires** (200 notés de façon fiable) : situation actuelle sur
14 domaines, 50 ans d'historique, indice d'avenir à l'horizon 2100, risque politique et pronostics électoraux,
rapports PDF sourcés. Python · Streamlit · HTML/CSS/JS · CSV · machine learning.

## Ce que fait l'application

- **Carte du monde** rouge / orange / vert (score global), cliquable, avec un sélecteur : situation actuelle,
  indice d'avenir, risque politique, chacun des 14 piliers, ou l'historique année par année depuis 1975.
- **Barre de recherche** : tapez un pays (français, anglais ou code ISO) → analyse complète rédigée :
  synthèse, SWOT, 14 domaines (constat chiffré, comparaison régionale, tendance sur 50 ans, projection 2050,
  lecture d'expert, liens vers les institutions de référence), prospective 2100, politique et élections,
  marchés financiers, actualités et déclarations, avis d'expert IA (optionnel), sources.
- **PDF** : rapport global (≈ 12-14 pages), rapport détaillé par critère (≥ 4 pages, séries longues et sources),
  rapport élections & risque politique.
- **Images** : affiche du pays (silhouette aux couleurs du score), radar, carte du monde en PNG.
- **Élections** : calendrier modifiable, saisie / import de sondages avec agrégation et probabilités
  (Monte-Carlo), indice de risque politique, simulateur « et si… ».
- **Bouton « Mettre à jour les informations »** : relance les agents, complète les CSV historiques,
  réentraîne le ML, recalcule la carte et (si configuré) enregistre le tout dans GitHub.

## Les agents

| Agent | Poids | Principales sources |
|---|---|---|
| 🩺 Santé | 8 | Banque mondiale (WDI) |
| 🎓 Éducation & PISA | 8 | OCDE PISA 2022, WDI |
| 📈 Économie & Macro | 9 | WDI |
| 💹 Marchés financiers & Finance | 5 | WDI, Yahoo Finance |
| 🏢 Business & Entrepreneuriat | 6 | WDI, IDEA |
| 🌍 Environnement & Climat | 7 | WDI |
| 🌾 Agriculture & Sécurité alimentaire | 6 | WDI |
| 🏛️ Gouvernance & Démocratie | 8 | V-Dem, EIU, IDEA |
| ⚖️ Justice & Sentiment d'impunité | 8 | V-Dem, IDEA, ONUDC/WDI, CPJ |
| 🕵️ Corruption & Intégrité | 7 | Transparency International, V-Dem |
| 🛡️ Sécurité & Paix | 9 | WDI, V-Dem, IDEA, HCR |
| 😊 Social & Bien-être | 8 | World Happiness Report, WDI |
| 🚀 Innovation & Technologie | 7 | WDI |
| ✈️ Tourisme & Attractivité | 4 | WDI |
| 🔭 Démographie & Prospective 2100 | — | ONU World Population Prospects + ML |
| 🗳️ Pronostics électoraux & risque politique | — | ML sur V-Dem + WDI, sondages |
| 📰 Veille actualités & déclarations | — | Google News, GDELT |
| 🧠 Synthèse & consensus | — | tous les CSV |

Méthode détaillée : [`docs/METHODOLOGIE.md`](docs/METHODOLOGIE.md).

## Démarrer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Les CSV fournis dans `data/` sont déjà calculés (collecte du 18/10/2026). Pour tout recalculer :

```bash
python -m core.pipeline                     # collecte + ML + scores
python -m core.pipeline --skip-fetch        # recalcul seul
python -m core.pipeline --with-markets --with-calendar
```

## Mettre en ligne (GitHub + Streamlit Community Cloud)

1. Créez un dépôt GitHub et poussez ce dossier.
2. Sur [share.streamlit.io](https://share.streamlit.io), créez une app pointant sur `app.py`.
3. Dans *App settings → Secrets*, collez le contenu de `.streamlit/secrets.toml.example` en le complétant :
   - `GITHUB_TOKEN`, `GITHUB_REPO`, `GITHUB_BRANCH` pour que le bouton de mise à jour enregistre les CSV
     dans le dépôt (le disque de Streamlit Cloud est effacé à chaque redémarrage) ;
   - une clé `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY` ou une URL `OLLAMA_URL` pour l'avis d'expert IA.
4. La GitHub Action `.github/workflows/update-data.yml` relance toute la collecte chaque lundi, y compris V-Dem
   (trop lourd en mémoire pour Streamlit Cloud : dans l'application, ses séries sont conservées telles quelles
   et rafraîchies par l'Action).

## Structure

```
app.py                     interface Streamlit (7 pages)
assets/style.css           identité visuelle
core/catalog.py            agents, indicateurs, poids, références d'experts
core/sources.py            accès aux données (GitHub open-numbers, API Banque mondiale, V-Dem)
core/agents.py             collecte indépendante par agent → data/agents/*.csv
core/scoring.py            normalisation, piliers, score actuel, historique 1970 →
core/ml_engine.py          projections 2050, indice d'avenir 2100, modèle de risque politique, pays comparables
core/elections.py          indice de risque, sondages, calendrier, simulateur
core/narrative.py          rédaction de l'analyse pays
core/llm.py                avis d'expert (Anthropic, Mistral ou Ollama)
core/markets.py, news.py   bourse, actualités
core/images.py             affiches, radars, cartes, graphiques
core/pdf_reports.py        rapports PDF
core/github_sync.py        sauvegarde des données dans le dépôt
core/pipeline.py           orchestrateur (bouton, Action GitHub, ligne de commande)
data/agents/               historique brut par agent (format long, 1970 → 2025, projections ONU → 2100)
data/scores/               scores actuels, historiques, instantanés de chaque mise à jour
data/ml/                   projections, indice d'avenir, risque politique, métriques d'apprentissage
data/elections/            calendrier et sondages (modifiables)
data/ref/                  référentiel pays, PISA, géométries
```

## Limites à connaître

Les scores reflètent les données publiques disponibles, avec leurs retards (souvent 1 à 3 ans) et leurs trous ;
la fiabilité est affichée pour chaque pays. Les projections sont des scénarios tendanciels. Les probabilités par
candidat n'existent que si des sondages ont été saisis. Les valeurs PISA de `data/ref/pisa.csv` ont été saisies
d'après le rapport OCDE PISA 2022 : vérifiez-les contre la publication officielle avant tout usage public, et
remplacez le fichier quand PISA 2025 sera publié.
