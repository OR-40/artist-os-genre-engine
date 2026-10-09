# Candidats de modèles — évaluation sans entraînement

Date de vérification des fiches : 2026-10-09.

## Règle de sécurité

Ces modèles sont des **candidats de test uniquement**. Leur licence publiée ne suffit pas à conclure que leurs données d'entraînement, leurs poids et tous les droits associés sont juridiquement sécurisés pour une exploitation commerciale. Aucun modèle ne doit être intégré à ARTIST OS ni proposé aux utilisateurs avant revue de licence et tests reproductibles.

## Candidat A — baseline de classification musicale

- Modèle : https://huggingface.co/dima806/music_genres_classification
- Licence indiquée sur la fiche : Apache-2.0.
- Base technique indiquée : wav2vec2.
- Entraînement déclaré : GTZAN, 1 000 extraits de 30 secondes, 10 classes (blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock).
- Intérêt : baseline directe de classification de genres, avec une taxonomie simple et connue.
- Limites : dix classes seulement ; les styles hybrides et la couleur artistique ne sont pas explicitement couverts par ces étiquettes. La fiche ne constitue pas une validation indépendante de la qualité sur des chansons complètes. Les droits et conditions du dataset d'entraînement et du modèle de base doivent être examinés avant toute exploitation commerciale.

## Candidat B — classifieur AST entraîné sur des mashups

- Modèle : https://huggingface.co/neerajs7/AST-audio-classifier
- Licence indiquée sur la fiche : MIT.
- Architecture déclarée : Audio Spectrogram Transformer (AST), environ 86,5 millions de paramètres entraînables.
- Entraînement déclaré : données de mashups pour 10 genres ; la fiche affiche des scores de compétition, qui ne doivent pas être interprétés comme une garantie de qualité sur les chansons réelles d'ARTIST OS.
- Intérêt : candidat à comparer pour les mélanges de genres.
- Limites : modèle relativement volumineux pour un service CPU ; la provenance/licence du dataset de mashups et les droits du modèle de base doivent être examinés avant tout usage commercial. Les performances revendiquées doivent être reproduites sur notre propre jeu de test.

## Décision technique provisoire

1. Commencer par le candidat A comme baseline, puis comparer le candidat B sur les mêmes fichiers si l'environnement le permet.
2. Utiliser les modèles uniquement en inférence pour cette première comparaison : pas de fine-tuning, pas de constitution de corpus d'entraînement à partir des pistes Incompetech.
3. Ne pas traiter les sorties du modèle comme vérité terrain. Deux personnes doivent confirmer les étiquettes de référence à l'écoute, en conservant les désaccords.
4. Tester des morceaux complets et des fenêtres réparties dans le temps ; agréger les prédictions sans laisser un seul court passage décider du genre global.
5. Inclure des cas difficiles : rock avec accents tribaux, rock/metal, pop-rock, hip-hop avec instruments rock, morceaux acoustiques/électriques et mélanges de styles.
6. Séparer les morceaux de test des morceaux utilisés pour régler les paramètres, et si possible séparer par artiste.
7. Mesurer macro-F1, précision/rappel par genre, matrice de confusion, durée totale d'analyse et mémoire maximale sur CPU.
8. Avant une décision de production, archiver les versions exactes des poids, leurs fichiers de licence, la licence du modèle de base, la provenance des données d'entraînement déclarée et les résultats de tests.

## Corpus

Aucun fichier audio n'a encore été téléchargé ni validé juridiquement dans le dépôt. Les 12 pistes d'Incompetech restent des candidates pending avec les deux permissions consignées unclear. Le premier test reproductible ne pourra commencer qu'avec des fichiers dont les droits d'utilisation prévus ont été vérifiés et une vérité terrain musicale documentée.
