### Sources

| Domaine | Producteurs primaires | Accès |
|---|---|---|
| Économie, santé, éducation, environnement, agriculture, finance, innovation, tourisme, sécurité | Banque mondiale — World Development Indicators | miroir `open-numbers` sur GitHub, repli API `api.worldbank.org` |
| Démocratie, état de droit, contrôle judiciaire, corruption, polarisation (1789 → 2025) | V-Dem Institute (Université de Göteborg), v16 | paquet officiel `vdeminstitute/vdemdata` |
| Perception de la corruption | Transparency International (CPI) | Gapminder Fast Track |
| Bonheur déclaré | World Happiness Report (Gallup) | Gapminder Fast Track |
| Démocratie (EIU), accès à la justice, administration impartiale (IDEA) | EIU, International IDEA | Gapminder Fast Track |
| Population et espérance de vie jusqu'en 2100 | Nations unies — World Population Prospects | Gapminder Fast Track |
| Compétences des élèves | OCDE — PISA 2022 | `data/ref/pisa.csv` |
| Bourse | Yahoo Finance (indices nationaux ou ETF pays) | `yfinance` |
| Actualités | Google News, GDELT | flux publics |

### Calcul des scores

1. Pour chaque indicateur : dernière valeur connue (au plus 8 ans).
2. Transformations : logarithme pour les grandeurs très dispersées (PIB par habitant, brevets…), distance à une cible
   (inflation et taux réel : 2 %), rapport à la population quand c'est pertinent.
3. Normalisation 0–100 entre le 5e et le 95e centile mondial, dans le sens favorable. Les mêmes bornes servent pour
   toutes les années : l'historique 1975 → aujourd'hui est comparable.
4. Pilier noté si au moins 35 % du poids de ses indicateurs est renseigné (PISA ne compte que pour les pays participants).
5. Situation actuelle = moyenne pondérée des 14 piliers (au moins 6). Fiabilité = part des piliers et indicateurs renseignés.
6. Score global = 75 % situation actuelle + 25 % indice d'avenir. Carte : vert ≥ 65, orange ≥ 45, rouge en dessous.

### Machine learning

- **Trajectoires 2035 / 2050** : tendance linéaire sur 25 ans pour chaque indicateur et chaque pays, amortie
  (demi-vie 15 ans), fourchette tirée de la dispersion historique → scénarios pessimiste, central, optimiste.
- **Indice d'avenir 2100** : démographie (évolution de la population ONU 2025 → 2100, fécondité, longévité),
  capital humain, institutions (niveau + tendance 20 ans), climat et ressources, dynamique économique.
- **Risque politique** : gradient boosting entraîné sur le panel pays × année (1980 → aujourd'hui) pour prédire un recul
  démocratique marqué dans les 3 ans (V-Dem). Validation temporelle (entraînement ≤ 2008, test ensuite) ; l'AUC est
  affichée et journalisée à chaque mise à jour dans `data/ml/metrics_log.csv`.
- **Pays comparables** : k-means et plus proches voisins sur les 14 piliers.

### Limites connues

- Les séries IDEA et EIU publiées par Gapminder s'arrêtent en 2020–2021 ; V-Dem (2025) porte l'essentiel du poids
  gouvernance / justice / corruption.
- Les arrivées touristiques de la Banque mondiale s'arrêtent en 2020.
- Taïwan et de nombreux territoires (outre-mer, petites îles) sont peu couverts par les bases internationales :
  leur fiabilité est affichée et ils restent en gris faute de données suffisantes.
- Les départements français d'outre-mer (Guadeloupe, Martinique, Réunion…) sont comptés avec la France dans ces bases.
- Doing Business a été arrêté en 2021 ; l'agent Business utilise la densité de créations d'entreprises, les IDE,
  le crédit, les infrastructures et la prévisibilité des règles. B-READY sera intégrable dès que la série sera complète.
- PISA 2025 sera publié fin 2026 : remplacez `data/ref/pisa.csv` (mêmes colonnes) pour l'intégrer.
- Les pronostics électoraux par candidat exigent des sondages : ils ne sont calculés que pour les élections dont
  les sondages ont été saisis ou importés.
