# ARTIST DNA — audit technique du 9 octobre 2026

## Ce qui est réellement établi

- Le décodage MP3 vers WAV mono PCM 16 kHz fonctionne sur la piste de test.
- FireRedVAD détecte des événements de chant. Cela ne signifie pas qu'il identifie le timbre, le registre ou le genre vocal.
- L'ancienne orchestration ne classait que trois extraits de 10 secondes sur une chanson de 224,5 secondes avec la baseline wav2vec2. Un résultat d'un seul passage pouvait donc peser autant que les autres, et l'entrée ne correspondait pas à la durée de 30 secondes des exemples GTZAN documentés pour la baseline.
- Le classifieur instrumental ONNX a été entraîné sur des clips de trois secondes centrés sur un seul instrument. Sa fiche indique que les mixages contenant plusieurs instruments peuvent être mal classés. Le score de violon observé à 0,2082 ne justifie donc pas de présenter le violon comme instrument dominant.

## Corrections apportées dans cette branche

1. Baseline wav2vec2 : jusqu'à huit fenêtres de 30 secondes réparties sur toute la chanson.
2. AST : checkpoint standard `Koras1k/ast-megafinetuned-gtzan-v2-0.97score`, avec artefacts Transformers publiés, fenêtres de 10 secondes et jusqu'à 24 fenêtres. L'ancien checkpoint `neerajs7/AST-audio-classifier` est écarté car son dépôt ne fournit pas les artefacts de configuration attendus par le chargeur générique.
3. Le champ `genres` repose sur des votes top-1 répétés à travers les fenêtres ; les prédictions par fenêtre restent consultables.
4. Le champ `instrumentation` ne reprend plus les candidats instrumentaux faibles. Les prédictions brutes sont conservées dans `instrument_analysis.predictions` et les éléments retenus dans `instrument_analysis.confirmed_predictions`.
5. `artistic_analysis` rédige un texte français à partir des indices disponibles et énonce les limites au lieu d'inventer un sous-genre, un timbre vocal ou des instruments.

## Ce qui n'est pas encore validé

- L'inférence AST n'a pas encore été exécutée sur la chanson réelle depuis cette branche. Le modèle publie des résultats sur GTZAN, ce qui ne garantit pas une bonne généralisation à une chanson indépendante.
- Le consensus entre baseline et AST n'a pas encore été mesuré sur un jeu de chansons annotées indépendamment.
- Le modèle ONNX d'instruments n'est pas validé sur des mixages polyphoniques complets. Les seuils de récurrence sont des garde-fous contre les affirmations faibles, pas une preuve de précision.
- FireRedVAD n'identifie pas le timbre vocal. Une analyse fiable du registre/timbre nécessiterait un modèle spécifique, idéalement sur une voix isolée ou une séparation vocale validée.
- Les deux modèles de genres n'annoncent que dix genres larges : ils ne suffisent pas à établir un sous-genre précis.
- Les licences affichées sur les poids ne prouvent pas à elles seules que les droits sur les jeux de données et les modèles de base couvrent un service commercial. Ce point reste bloquant avant commercialisation.
- GitHub Actions a exécuté avec succès les tests unitaires et de contrat sur le commit de cette branche : 64 tests passés. Cette suite n'effectue pas de téléchargement ni d'inférence des modèles et ne constitue donc pas une validation audio réelle.

## Référence des modèles

- Baseline : https://huggingface.co/dima806/music_genres_classification
- AST chargé par cette branche : https://huggingface.co/Koras1k/ast-megafinetuned-gtzan-v2-0.97score
- Ancien checkpoint AST exclu : https://huggingface.co/neerajs7/AST-audio-classifier
- Instrument ONNX : https://huggingface.co/onnx-community/Musical-Instrument-Classification-ONNX

**Décision :** cette branche améliore la couverture temporelle et réduit les faux positifs exposés à ARTIST OS. Elle n'est pas encore une validation de qualité commerciale et ne doit pas être fusionnée en production sur la seule base des tests unitaires.
