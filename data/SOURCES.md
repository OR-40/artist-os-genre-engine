# Évaluation initiale des sources de musique

Date de vérification : 2026-10-09.

Ce document évite de télécharger un corpus simplement parce qu'il est présenté comme « libre ». Une source n'est admise qu'après vérification des conditions de chaque piste et de l'usage prévu.

## Candidats examinés

### MTG-Jamendo — refusé pour le pilote commercial sans autorisation écrite

Le dépôt du dataset précise que son usage est limité à la recherche non commerciale et à l'usage académique ; toute application commerciale nécessite une autorisation écrite préalable de Jamendo. Il ne doit donc pas être utilisé pour entraîner ARTIST OS sans cette autorisation.

Source : https://github.com/MTG/mtg-jamendo-dataset

### Free Music Archive — pas d'aspiration automatisée

La licence est attachée à chaque morceau et doit être vérifiée sur la page correspondante. Les conditions du site interdisent le data mining. Ne pas télécharger en masse ni utiliser un dump du corpus sans base contractuelle adaptée. Une piste individuelle peut être examinée manuellement, mais seulement si sa licence exacte et les droits pertinents sont établis.

Sources :
- https://freemusicarchive.org/License_Guide
- https://freemusicarchive.org/terms_of_use

### SoundSafari/CC0-1.0-Music — non admis automatiquement

Le dépôt agrège des milliers de morceaux annoncés CC0 provenant de plusieurs sites. Cette déclaration de corpus n'est pas une preuve suffisante pour chaque fichier, et des questions ont été soulevées publiquement sur certains fichiers dont les sources d'origine auraient des restrictions de réutilisation. Ne pas ingérer ce dépôt en bloc. Une piste ne peut être considérée qu'après contrôle de sa page source d'origine, de sa licence et des éventuelles restrictions de la plateforme.

Source : https://github.com/SoundSafari/CC0-1.0-Music

### Keynata Commons — adapté seulement à un test technique préliminaire

Le projet annonce des compositions générées par un moteur à règles et publiées sous CC0, avec catégories telles que rock, J-pop et classique. Cela pourrait servir à vérifier le chargement audio, les métadonnées et le pipeline. Mais des morceaux synthétiques/générés ne sont pas représentatifs de la musique réelle d'artistes indépendants ; ils ne doivent pas constituer la preuve de qualité du futur classificateur ni être le seul corpus d'entraînement.

Source : https://carf-coder.github.io/keynata-commons/

## Décision

Aucune de ces sources n'est importée automatiquement dans `data/audio/`. Le premier lot de référence doit être constitué manuellement, piste par piste, avec la preuve de licence conservée et une étiquette de genre confirmée par écoute. Les sources CC0 annoncées par un agrégateur restent des candidates à vérifier, pas des données approuvées par défaut.

Pour le pilote, séparer clairement :
1. **test technique** : vérifier qu'un fichier est lisible et que le pipeline fonctionne ;
2. **évaluation musicale** : tester sur de vrais morceaux représentatifs, dont les droits sont vérifiés ;
3. **entraînement** : seulement après une revue des droits et des labels, et après avoir défini une séparation par artiste entre apprentissage et test.
