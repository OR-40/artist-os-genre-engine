# Déploiement ARTIST DNA CPU sur o2switch — procédure de préproduction

Cette procédure prépare un hébergement CPU sans nouveau fournisseur ni GPU. Elle n'effectue aucun déploiement et ne modifie pas le service paroles existant.

## Architecture

- Garder le service paroles actuel sur son domaine et son chemin existants.
- Créer une application Python distincte, idéalement sur un sous-domaine audio dédié (par exemple un sous-domaine choisi dans cPanel, sans remplacer le service actuel).
- L'application accepte **uniquement les fichiers MP3** via `POST /dna/analyze`, champ multipart `audio`. Le MP3 est décodé en WAV PCM mono 16 kHz temporaire pour les modèles qui exigent ce format ; il n'est pas réencodé en MP3.
- Le proxy Cloudflare ARTIST OS transmet `Authorization: Bearer <DNA3_TOKEN>`.
- Le moteur exige désormais le secret serveur `ARTIST_DNA_API_TOKEN` et refuse les requêtes d'analyse sans jeton. Si le secret n'est pas configuré, les routes d'inférence répondent HTTP 503 (fail closed).

## Prérequis

1. Vérifier dans cPanel que « Setup Python App » est disponible et choisir Python 3.12.
2. Créer l'application dans un répertoire racine **distinct du document root public**. Ne pas placer le code, les poids ou l'environnement virtuel dans le répertoire public du domaine.
3. Le décodage MP3 utilise `miniaudio`, une bibliothèque embarquée dans l'environnement Python ; FFmpeg n'est plus un prérequis. Après installation des dépendances, vérifier que `mp3_decoder_available` vaut `true` dans `/health`.
4. Vérifier les limites CPU, mémoire et temps d'exécution du compte. L'analyse d'un morceau complet peut être longue ; ne pas annoncer le service comme opérationnel avant d'avoir mesuré une requête réelle et vérifié l'absence de timeout Passenger.

## Fichiers et lancement

Déployer le contenu de la branche de préparation `fix/secure-api-token-o2switch` dans le répertoire de l'application. Le fichier de démarrage Passenger est `passenger_wsgi.py`, avec le point d'entrée `application`.

Installer les dépendances depuis l'environnement virtuel créé par cPanel :

```bash
pip install -r requirements-o2switch.txt
```

Le fichier de dépendances sélectionne la distribution PyTorch CPU et n'active pas de GPU. Si le gestionnaire de paquets choisit une roue CUDA ou échoue sur la résolution PyTorch, arrêter : ne pas remplacer par une installation GPU ou un service payant.

## Variables secrètes côté hébergeur

Créer ces variables dans la configuration de l'application Python ; ne jamais les écrire dans Git ni les transmettre dans le chat :

- `ARTIST_DNA_API_TOKEN` : secret aléatoire fort, partagé avec le secret Cloudflare `DNA3_TOKEN`.
- `HF_TOKEN` : jeton Hugging Face avec accès de lecture au dépôt privé `OR-40/artist-os-ast-onnx`.
- `ARTIST_DNA_ONNX_GENRE_ENABLED=true`.
- `ARTIST_DNA_ONNX_GENRE_REPO_ID=OR-40/artist-os-ast-onnx`.
- `ARTIST_DNA_INSTRUMENTS_ENABLED=false` pour la première recette CPU ; les instruments restent un module expérimental qu'on pourra tester séparément.

Les poids sont téléchargés côté serveur à la première inférence et mis en cache dans le répertoire utilisateur. Le jeton Hugging Face ne doit jamais être transmis au navigateur.

## Tests avant de brancher ARTIST OS

1. `GET /health` doit répondre 200 et confirmer `mp3_decoder_available: true`.
2. `POST /dna/analyze` sans jeton doit répondre 401.
3. Avec un jeton invalide, la route doit répondre 401.
4. Avec le jeton correct et un petit MP3 de test, l'API doit répondre `ok: true`, inclure `genre_analysis.models.genre_candidate_onnx`, `audio_tag_analysis` et `vocal_analysis`.
5. Tester ensuite le morceau de référence complet, mesurer la durée et vérifier la consommation CPU/RAM. Ne pas confondre le smoke test synthétique avec une validation de précision musicale.

## Branchement Cloudflare

Après réussite des tests précédents seulement, configurer **Preview uniquement** dans Cloudflare Pages :

- `DNA_URL` = URL HTTPS de l'application audio dédiée, sans chemin `/dna/analyze` (le proxy l'ajoute).
- `DNA3_TOKEN` = même valeur que `ARTIST_DNA_API_TOKEN`.

Ne pas modifier les secrets Production avant une recette complète de Preview. Le service paroles peut continuer d'utiliser son URL existante ; la passerelle unifiée peut aussi utiliser la même base URL si le service accepte le champ `lyrics`.

## Critère d'arrêt

Si l'installation des dépendances CPU, le téléchargement des poids privés, le décodage MP3, la mémoire ou le délai de réponse échoue sur l'offre o2switch actuelle, ne pas activer la route et ne pas acheter de ressource. Conserver la passerelle en 503 et reprendre le choix d'hébergement à partir des mesures observées.
