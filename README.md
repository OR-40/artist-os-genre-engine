# ARTIST OS — Genre Engine

Prototype isolé destiné à évaluer une analyse de genres musicaux peu coûteuse, avant toute intégration à ARTIST OS.

**Ce dépôt ne déploie rien en production et ne modifie pas le site ARTIST OS.** Il contient un validateur de manifeste, un prototype de détection d'événements vocaux FireRedVAD et une API locale de test. Le candidat `msaligs/ast_fine_tuned_music_genre_10` a donné un signal rock plus cohérent sur un morceau de test, mais il n'est pas encore validé sur un jeu de référence.

## Moteur ARTIST DNA — prototype d'orchestration

Le moteur CPU isolé est en cours de validation sur la branche `feat/ast-msaligs-candidate-main` :

- `POST /dna/analyze` combine deux candidats de classification de genres (baseline wav2vec2 et candidat AST) avec la détection vocale FireRedVAD.
- Les deux classifieurs sont chargés à la demande. Les identifiants peuvent être remplacés avec `ARTIST_DNA_MODEL_BASELINE` et `ARTIST_DNA_MODEL_AST`.
- Pour la classification des genres, le moteur utilise la piste entière si elle dure au plus 30 secondes ; sinon, il sélectionne trois fenêtres de 10 secondes au début, au milieu et à la fin (30 secondes au total). Le candidat ONNX reçoit le même audio sélectionné. Les scores ne sont pas présentés comme des probabilités calibrées. Cette stratégie accélère l'analyse mais peut manquer un instrument ou un changement présent ailleurs dans le morceau.
- L'analyse instrumentale par défaut utilise `src/instrument_classifier.py` : modèle ONNX quantifié `onnx-community/Musical-Instrument-Classification-ONNX`, exécution locale via `CPUExecutionProvider`, jusqu'à six extraits de 3 secondes. Le fichier ONNX quantifié fait environ 90 Mo et est téléchargé depuis Hugging Face au premier usage, puis mis en cache.
- `ARTIST_DNA_INSTRUMENTS_ENABLED=true` active l'analyse ONNX par défaut ; définir explicitement `false` permet de la désactiver. Aucun service distant d'instruments ni variable `INSTRUMENTS_URL` n'est requis par défaut.
- La signature artistique reste volontairement non générée tant que les preuves musicales ne suffisent pas. Les prédictions d'instruments sont expérimentales : le modèle cible neuf catégories et peut confondre des instruments dans un mix complet.
- Les deux classifieurs de genres restent actifs : baseline wav2vec2 et candidat AST, tous deux exécutés sur CPU via Transformers (`device=-1`). Le modèle AST compte environ 86,5 millions de paramètres ; la latence et la mémoire doivent être mesurées sur l'hébergement cible avant toute mise en production.
- Si l'analyse ONNX échoue, l'orchestrateur conserve l'analyse des genres et de la voix, et signale l'analyse instrumentale comme indisponible. Les erreurs sont journalisées sans inclure le contenu audio.
- `src/remote_instruments.py` reste disponible pour les tests et une compatibilité explicite, mais n'est plus utilisé par défaut.
- Le pipeline complet est défini par `requirements-dna.txt`. L'API MP3 nécessite l'exécutable système FFmpeg pour le décodage temporaire ; les tests de contrat simulent ce décodage.
- Endpoint prototype `POST /analyze` : réponse `{ "analysis": ... }`. Endpoint orchestrateur `POST /dna/analyze` : champ multipart `audio` (alias `file` conservé pour tests directs) et réponse `{ "ok": true, "dna": ... }`, compatible avec le proxy `/api/dna3` existant dans ARTIST OS. Les deux acceptent un MP3 de 50 Mio maximum, décodé en WAV PCM mono 16 kHz temporaire avec FFmpeg.
- WAV/FLAC/OGG/M4A refusés à l'entrée : seul le MP3 est accepté. Le décodage interne dépend de l'exécutable FFmpeg disponible sur le serveur ; aucun déploiement ni branchement production.
- Pas de téléchargement automatique de musique.
- Premier inventaire de 12 morceaux enregistrés de Kevin MacLeod, avec pages officielles et licence CC BY 4.0 consignées dans `data/pilot_candidates.csv`.
- Ces 12 pistes restent `pending` : aucun audio n'a été téléchargé et elles ne sont pas encore admises à l'entraînement/évaluation officielle.
- Pas de dataset audio réel validé pour l'entraînement.

### Exécution du candidat AST exporté en ONNX

Le candidat AST exporté s'exécute directement via ONNX Runtime sur CPU, sans charger les poids PyTorch. Le fichier de poids (environ 345 Mo) est conservé hors de Git dans le dépôt privé Hugging Face `OR-40/artist-os-ast-onnx`.

**Mode recommandé pour le runtime isolé :**
- Activer explicitement l'adaptateur en définissant `ARTIST_DNA_ONNX_GENRE_REPO_ID=OR-40/artist-os-ast-onnx` (ou `ARTIST_DNA_ONNX_GENRE_ENABLED=true`).
- Ajouter `HF_TOKEN` comme secret côté serveur, avec uniquement un droit de lecture sur le dépôt privé du modèle. Ne jamais le committer ni l'afficher dans les logs.
- Une fois l'adaptateur activé, le moteur télécharge et met en cache uniquement `model.onnx`, `config.json`, `preprocessor_config.json` et `genre_labels.json` depuis `ARTIST_DNA_ONNX_GENRE_REPO_ID` (par défaut `OR-40/artist-os-ast-onnx`).
- Le cache Hugging Face évite de retélécharger le modèle à chaque chargement si le disque de l'environnement est persistant.
- Pour un environnement qui fournit déjà les fichiers, définir `ARTIST_DNA_ONNX_GENRE_MODEL_PATH` sur le dossier local ; ce chemin explicite prend priorité sur le téléchargement distant.
- Le dépôt de modèle est privé. Sans `HF_TOKEN` ou avec un jeton sans droit de lecture, le chargement échoue explicitement. Le jeton n'est pas inclus dans les messages d'erreur.

Le téléchargement à froid nécessite une connexion réseau et environ 345 Mo d'espace disque. Avant un déploiement, il faut vérifier la persistance du cache, les limites de mémoire/CPU et la latence de l'API sur l'hébergement cible.

Le classifieur vérifie le nom d'entrée ONNX, le nombre de sorties et la présence des labels. Les labels sont explicitement mappés dans `genre_labels.json`; cette correspondance reflète l'ordre documenté du candidat et doit rester traçable. Les scores sont des sorties du classifieur, pas des probabilités calibrées.

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

## Préparation d'hébergement CPU sur o2switch

Une branche de préparation ajoute un point d'entrée Passenger WSGI et protège les routes d'inférence par `Authorization: Bearer`. Les routes d'analyse refusent de fonctionner si `ARTIST_DNA_API_TOKEN` n'est pas défini.

Procédure de préproduction (sans toucher au service paroles actuel ni à ARTIST OS Production) : [docs/O2SWITCH-CPU-DEPLOYMENT.md](docs/O2SWITCH-CPU-DEPLOYMENT.md).

Cette préparation n'est pas un déploiement : l'installation des dépendances, FFmpeg, le temps de réponse et la mémoire doivent encore être validés sur l'hébergement réel avant de renseigner `DNA_URL`.
