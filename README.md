# ARTIST OS — Genre Engine

Prototype CPU isolé de l'orchestration ARTIST DNA. L'intégration technique est testée en CI avec un MP3 synthétique et le modèle AST ONNX privé, mais la précision musicale n'est pas validée sur un corpus de référence ni sur tous les morceaux de l'artiste.

**Ce dépôt ne déploie rien en production et ne modifie pas le site ARTIST OS.** Les poids restent hors de Git ; le moteur n'est pas déclaré opérationnel tant qu'un vrai MP3 n'a pas été traité sur l'hébergement cible et dans ARTIST OS Preview.

## Moteur ARTIST DNA — orchestration CPU

- `POST /dna/analyze` combine un classifieur dédié de genres, le tagueur AST AudioSet (séparé des genres) et FireRedVAD. Le classifieur de genre ONNX peut utiliser le dépôt privé `OR-40/artist-os-ast-onnx`; sans activation ONNX, le candidat `msaligs/ast_fine_tuned_music_genre_10` est utilisé via Transformers sur CPU.
- Pour les morceaux longs, le moteur choisit jusqu'à dix fenêtres de 10 secondes réparties dans la piste. L'analyse auxiliaire AST AudioSet et les instruments utilisent un sous-ensemble de fenêtres. Cela réduit la latence mais peut manquer un élément présent ailleurs dans le morceau.
- Les scores de genre sont des sorties de modèles, pas des probabilités calibrées. Les tags AST AudioSet sont des événements/classes audio et ne sont pas mélangés au consensus des genres.
- La détection vocale repose sur FireRedVAD. L'analyse instrumentale utilise un classifieur ONNX expérimental de neuf catégories, exécuté localement sur CPU ; elle peut confondre les instruments dans un mix complet. Pour la première recette d'hébergement, la désactiver avec `ARTIST_DNA_INSTRUMENTS_ENABLED=false`.
- La signature artistique n'est pas générée à partir des seuls scores de genre ; elle exige des preuves musicales suffisantes.
- Le modèle ONNX d'instruments et le modèle AST de genre sont téléchargés côté serveur et mis en cache. Les poids privés exigent un secret Hugging Face de lecture, jamais exposé au navigateur.
- Les fichiers audio sont décodés en WAV PCM mono 16 kHz temporaire par FFmpeg puis supprimés avec le répertoire temporaire. La taille maximale est de 50 Mio ; les formats MP3, WAV, FLAC, OGG et M4A sont acceptés.
- Les routes d'inférence `POST /analyze` et `POST /dna/analyze` exigent `Authorization: Bearer`. Le secret serveur `ARTIST_DNA_API_TOKEN` est obligatoire : s'il manque, l'API échoue en HTTP 503 ; un jeton absent ou incorrect reçoit HTTP 401. `GET /health` reste public et ne charge pas les poids.

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
