# ARTIST OS — Genre Engine

Prototype isolé destiné à évaluer une analyse de genres musicaux peu coûteuse, avant toute intégration à ARTIST OS.

**Ce dépôt ne déploie rien en production et ne modifie pas le site ARTIST OS.** Il contient un validateur de manifeste, un prototype de détection d'événements vocaux FireRedVAD et une API locale de test. Aucun modèle de classification de genres n'est encore validé.

## Moteur ARTIST DNA — prototype d'orchestration

Le moteur CPU isolé est en cours de validation sur la branche `fix/local-cpu-dna-fire-ast-onnx` :

- `POST /dna/analyze` combine la baseline de genres `dima806/music_genres_classification`, la détection d'événements vocaux FireRedVAD et le candidat instrumental ONNX local.
- Par défaut, seul le classifieur de genres baseline est actif. Le candidat AST `neerajs7/AST-audio-classifier` reste désactivé : le dépôt publié ne fournit pas le `config.json` attendu par le chargeur générique et son chargement dédié n'a pas encore été validé. Ne pas l'activer avec `ARTIST_DNA_ENABLE_AST=true` avant qu'un chargeur soit testé de bout en bout.
- Pour les genres, les pistes de plus de 30 secondes sont échantillonnées avec jusqu'à huit fenêtres de 30 secondes réparties du début à la fin. Cela correspond mieux à la durée des extraits du dataset GTZAN qu'utilise la baseline que les anciennes fenêtres de 10 secondes. Le moteur conserve les prédictions par fenêtre et ne promeut dans `genres` que les étiquettes qui arrivent régulièrement en tête. Les scores ne sont pas des probabilités calibrées et cette modification réduit l'effet d'un passage isolé sans garantir la justesse du genre.
- L'analyse instrumentale actuelle utilise `src/instrument_classifier.py` : modèle ONNX quantifié `onnx-community/Musical-Instrument-Classification-ONNX`, CPU local, jusqu'à six extraits de 3 secondes. Le modèle est annoncé comme entraîné sur des extraits centrés sur un seul instrument ; il n'est donc pas validé pour reconnaître les instruments dominants d'un mixage complet.
- Les prédictions instrumentales brutes sont conservées dans `instrument_analysis.predictions` pour diagnostic. Le champ de compatibilité `instrumentation` n'expose que les candidats dépassant des seuils de score et de récurrence ; ces seuils sont un garde-fou, pas une validation scientifique du modèle. Aucun service distant ni variable `INSTRUMENTS_URL` n'est requis.
- `artistic_analysis` produit un texte français fondé sur les indices disponibles et explicite ce qui n'est pas confirmé. La signature artistique complète reste non générée : le moteur ne prétend pas connaître un sous-genre, le timbre vocal ou des rôles instrumentaux sans modèles validés.
- La baseline de genres est exécutée sur CPU via Transformers (`device=-1`). Le modèle AST compte environ 86,5 millions de paramètres mais reste désactivé par défaut jusqu'à ce que son chargement dédié soit implémenté et validé ; la latence et la mémoire devront ensuite être mesurées sur l'hébergement cible.
- Si l'analyse ONNX échoue, l'orchestrateur conserve l'analyse des genres et de la voix, et signale l'analyse instrumentale comme indisponible. Les erreurs sont journalisées sans inclure le contenu audio.
- `src/remote_instruments.py` reste disponible pour les tests et une compatibilité explicite, mais n'est plus utilisé par défaut.
- Le pipeline complet est défini par `requirements-dna.txt`. L'API MP3 nécessite l'exécutable système FFmpeg pour le décodage temporaire ; les tests de contrat simulent ce décodage.
- Endpoint prototype `POST /analyze` : réponse `{ "analysis": ... }`. Endpoint orchestrateur `POST /dna/analyze` : champ multipart `audio` (alias `file` conservé pour tests directs) et réponse `{ "ok": true, "dna": ... }`, compatible avec le proxy `/api/dna3` existant dans ARTIST OS. Les deux acceptent un MP3 de 50 Mio maximum, décodé en WAV PCM mono 16 kHz temporaire avec FFmpeg.
- WAV/FLAC/OGG/M4A refusés à l'entrée : seul le MP3 est accepté. Le décodage interne dépend de l'exécutable FFmpeg disponible sur le serveur ; aucun déploiement ni branchement production.
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
