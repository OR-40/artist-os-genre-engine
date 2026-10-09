# Évaluation initiale des sources de musique

Date de vérification : 2026-10-09.

Ce document distingue les licences publiées des autorisations effectivement validées pour le projet. Une licence CC BY peut autoriser des réutilisations commerciales, mais impose l'attribution. Le statut `pending` signifie que la revue humaine finale du périmètre d'utilisation et de la provenance n'est pas terminée.

## Nouveau lot candidat : Incompetech / Kevin MacLeod

Douze enregistrements ont été inscrits dans `data/pilot_candidates.csv` comme **candidats réels**, sans téléchargement audio ni admission à l'entraînement à ce stade. Les pages individuelles indexées affichent une licence Creative Commons Attribution 4.0 (CC BY 4.0), et la page officielle destinée aux agents indique que le catalogue utilise cette licence et fournit les métadonnées de chaque piste. Pour éviter toute fausse validation, les deux champs d'autorisation (`training_use_permission` et `commercial_use_permission`) sont consignés `unclear` jusqu'à ce qu'une revue documentée confirme que la licence couvre bien l'entraînement du modèle et l'exploitation commerciale envisagée.

Sources officielles :
- Catalogue et métadonnées : https://incompetech.com/agent-section/
- Catalogue des pistes : https://incompetech.com/music/royalty-free/music.html
- Licences : https://incompetech.com/music/royalty-free/licenses/
- Texte CC BY 4.0 : https://creativecommons.org/licenses/by/4.0/

Pistes retenues pour examen individuel :
- Rock : Big Rock, Aitech, Exhilarate.
- Jazz : Study And Relax, Faster Does It.
- Funk : Enter the Party, Style Funk.
- Pop : Clear Air.
- Electronica : Equatorial Complex, Cloud Dancer.
- World : Cretaceous Dawn, Sauropod Spotting.

**Limites importantes :** il s'agit d'un premier lot de métadonnées de morceaux enregistrés d'un seul catalogue, pas d'un corpus représentatif de la diversité des artistes indépendants. Les pages fournissent une indication de licence et des étiquettes de genre éditoriales, pas une validation indépendante par écoute. L'utilisation pour l'entraînement commercial doit conserver les attributions et respecter CC BY 4.0. Avant d'approuver les fichiers audio, il faut confirmer que la licence couvre les usages précis prévus, que les droits pertinents sur chaque enregistrement sont couverts, et comment l'attribution sera conservée dans les artefacts et la documentation du projet.

Tous les morceaux restent `pending`. Aucun n'est éligible à l'entraînement ou à l'évaluation officielle tant que la revue humaine n'est pas enregistrée. Aucun fichier audio n'a été téléchargé ou ajouté au dépôt.

## Autres sources examinées

### MTG-Jamendo — refusé pour le pilote commercial sans autorisation écrite

Le dépôt du dataset précise que son usage est limité à la recherche non commerciale et à l'usage académique ; toute application commerciale nécessite une autorisation écrite préalable de Jamendo.

Source : https://github.com/MTG/mtg-jamendo-dataset

### Free Music Archive — pas d'aspiration automatisée

La licence est attachée à chaque morceau et doit être vérifiée sur la page correspondante. Les conditions du site interdisent le data mining. Ne pas télécharger en masse ni utiliser un dump du corpus sans base contractuelle adaptée.

Sources :
- https://freemusicarchive.org/License_Guide
- https://freemusicarchive.org/terms_of_use

### SoundSafari/CC0-1.0-Music — non admis automatiquement

Le dépôt agrège des milliers de morceaux annoncés CC0 provenant de plusieurs sites. Cette déclaration de corpus n'est pas une preuve suffisante pour chaque fichier. Ne pas ingérer ce dépôt en bloc.

Source : https://github.com/SoundSafari/CC0-1.0-Music

### Keynata Commons — adapté seulement à un test technique préliminaire

Le projet annonce des compositions générées par un moteur à règles et publiées sous CC0. Elles peuvent aider à vérifier un pipeline, mais ne sont pas représentatives de la musique réelle d'artistes indépendants.

Source : https://carf-coder.github.io/keynata-commons/

## Décision opérationnelle

1. Le premier lot candidat est listé, avec les URL de source et de licence par piste.
2. Les fichiers audio ne sont pas encore téléchargés.
3. Les 12 lignes restent en `pending` : aucun entraînement ne doit démarrer à partir de ce lot.
4. Après revue humaine, télécharger uniquement les pistes approuvées, conserver la preuve de licence et l'attribution, puis confirmer les genres par écoute.
5. Séparer les données par artiste lors de l'évaluation. Ce lot d'un seul catalogue ne suffit pas à démontrer la généralisation à d'autres artistes.
