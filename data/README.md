# Politique d'admission des données audio

## Statuts

- `pending` : droits non vérifiés ; piste exclue de tout entraînement et de toute évaluation officielle.
- `approved` : une personne a documenté une vérification humaine ; la piste est seulement candidate à l'usage autorisé par les conditions vérifiées.
- `rejected` : piste exclue.

Le statut ne prouve pas à lui seul qu'une piste est juridiquement utilisable. Conserver dans `notes` les restrictions, l'attribution requise et les preuves utiles, sans remplacer les champs de provenance.

## Règles de sélection

- Priorité aux enregistrements dont les droits sont clairement établis pour l'usage commercial envisagé.
- Vérifier les droits de l'enregistrement sonore **et** les droits sur la composition/interprétation lorsque c'est pertinent ; une licence de fichier ne garantit pas toujours que tous les droits nécessaires sont couverts.
- Les licences CC BY/CC0 peuvent permettre certains usages commerciaux, sous conditions propres à la licence ; vérifier la page de la piste et les obligations d'attribution.
- Exclure par défaut les licences contenant `NC` (NonCommercial), les licences `ND` lorsque l'usage envisagé peut être considéré comme une adaptation, les licences inconnues et les autorisations ambiguës, sauf autorisation écrite adaptée.
- Ne pas collecter en masse depuis un site si ses conditions interdisent l'extraction automatisée ou le data mining.
- Ne jamais enregistrer dans le dépôt Git les fichiers audio bruts. Le dossier `data/audio/` est ignoré par Git ; garder les fichiers dans un stockage local contrôlé.

## Pilote recommandé

Commencer par un petit lot de morceaux complets dont les droits ont été vérifiés un par un. Étiqueter chaque morceau à partir de l'écoute et d'informations fiables, puis faire une revue indépendante des genres. Séparer les données par artiste lors du test pour éviter qu'un même artiste ou morceau soit présent dans les deux ensembles.

Ne pas entraîner de modèle avec le manifeste d'exemple, des lignes `pending` ou `rejected`, ni avec une piste dont les droits restent incertains.
