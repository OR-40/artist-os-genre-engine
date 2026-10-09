# ARTIST OS — Genre Engine

Prototype isolé destiné à évaluer une analyse de genres musicaux peu coûteuse, avant toute intégration à ARTIST OS.

**Ce dépôt ne déploie rien en production et ne modifie pas le site ARTIST OS.** Il contient un validateur de manifeste, un prototype de détection d'événements vocaux FireRedVAD et une API locale de test. Aucun modèle de classification de genres n'est encore validé.

## Moteur ARTIST DNA — prototype d'orchestration

Une première passerelle isolée est disponible sur la branche `feat/artist-dna-orchestrator` :

- `POST /dna/analyze` combine deux candidats de classification de genres (baseline wav2vec2 et candidat AST) avec la détection vocale FireRedVAD.
- Les deux classifieurs sont chargés à la demande. Les identifiants peuvent être remplacés avec `ARTIST_DNA_MODEL_BASELINE` et `ARTIST_DNA_MODEL_AST`.
- Le moteur découpe l'audio en fenêtres de 30 secondes, conserve les prédictions par modèle et expose une agrégation explicite. Les scores ne sont pas présentés comme des probabilités calibrées.
- La signature artistique et l'analyse des instruments restent volontairement non connectées tant qu'une solution locale au prototype n'a pas été validée. Aucune dépendance Railway ne doit être utilisée.
- Dépendances du moteur combiné : `python -m pip install -r requirements-dna.txt`.
- Format d'entrée encore limité à WAV, 50 Mio maximum. Cette branche n'est pas déployée en production et n'est pas encore validée avec les poids réels ensemble.

Ce prototype est une passerelle logicielle, pas encore une validation de qualité, de latence, de mémoire, de licences ou de capacité multi-utilisateur. Ne pas le brancher au site ARTIST OS avant la revue des poids, le test contractuel et les mesures de déploiement.

## État actuel

- Validation de la structure du manifeste CSV à 14 colonnes.
- Contrôles des champs obligatoires, URL HTTP(S), identifiants dupliqués et statuts de revue.
- Permissions distinctes consignées pour l'entraînement et l'usage commercial.
- Export séparé des seules pistes marquées approved avec les deux permissions à yes, un réviseur et une date.
- Tests automatiques exécutés par GitHub Actions, y compris des tests de contrat de l'API locale.
- Endpoint prototype `POST /analyze` : réponse `{ "analysis": ... }` ; format accepté à ce stade : WAV uniquement.
- FLAC/OGG/MP3/M4A non pris en charge par l'API prototype tant qu'un décodeur dédié n'a pas été validé ; aucun déploiement ni branchement production.
- Pas de téléchargement automatique de musique.
- Premier inventaire de 12 morceaux enregistrés de Kevin MacLeod, avec pages officielles et licence CC BY 4.0 consignées dans `data/pilot_candidates.csv`.
- Ces 12 pistes restent `pending` : aucun audio n'a été téléchargé et elles ne sont pas encore admises à l'entraînement/évaluation officielle.
- Pas de dataset audio réel validé pour l'entraînement.

## Environnement

Python 3.12 ou version compatible. Le validateur n'a pas de dépendance externe.

Installer les dépendances légères de test (sans télécharger FireRedVAD ni ses poids) :

```bash
python -m pip install -r requirements-test-api.txt
```

Exécuter les tests :

```bash
python -m unittest discover -s tests -v
```

Créer un manifeste vierge :

```bash
python -m src.license_manifest template data/manifest.csv
```

Valider un manifeste :

```bash
python -m src.license_manifest validate data/manifest.csv
```

Exporter les lignes ayant passé les contrôles de structure et portant les deux permissions humaines explicites :

```bash
python -m src.license_manifest eligible data/manifest.csv --output data/eligible_manifest.csv
```

L'export est un garde-fou de workflow, pas une décision juridique. Il ne vérifie pas lui-même la validité d'une licence ni l'autorité de la personne qui a renseigné le CSV. Le fichier source reste intact et l'export peut contenir zéro piste.

Le fichier data/manifest.example.csv est fictif. Il sert uniquement à illustrer le format et **ne constitue pas une donnée d'entraînement**.

## Candidats de modèles

Deux modèles sont documentés pour une comparaison d'inférence, sans entraînement ni intégration en production : voir [docs/MODEL_CANDIDATES.md](docs/MODEL_CANDIDATES.md). Leurs licences de poids ne constituent pas une validation des droits sur leurs données d'entraînement ; aucune décision de production n'est prise avant revue juridique et tests sur des morceaux autorisés.

## Conditions avant toute constitution du dataset

1. Vérifier la source et les droits de chaque enregistrement individuellement.
2. Confirmer explicitement que les droits couvrent l'usage envisagé, y compris l'entraînement/évaluation d'un système et l'exploitation commerciale du service.
3. Vérifier les droits pertinents sur l'enregistrement et sur l'œuvre musicale, ainsi que les conditions attachées à toute licence ou autorisation.
4. Conserver la page source, la licence exacte, la date de vérification et la personne ayant vérifié les droits.
5. Écarter les pistes sous licence non commerciale, les licences incertaines et toute piste sans provenance vérifiable.
6. Ne pas contourner les conditions d'un site ni aspirer massivement ses fichiers.
7. Ne pas considérer le statut approved du CSV comme une validation juridique automatique : c'est une trace de revue humaine, pas un avis juridique.

## Protocole d'évaluation visé

Le pilote devra d'abord utiliser un petit ensemble contrôlé de morceaux complets, avec étiquettes confirmées manuellement et diversité d'artistes. Il faudra mesurer les erreurs par genre et tester sur des artistes absents de l'ensemble de référence. Les extraits d'un même morceau ne doivent pas se retrouver à la fois dans les ensembles de référence et de test.

Aucune intégration ARTIST OS ne doit être faite avant que la provenance des données, la qualité des résultats, la latence et la mémoire CPU soient documentées.
