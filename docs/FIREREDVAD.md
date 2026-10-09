# FireRedVAD — vocal-event detection prototype

This module is an isolated prototype for detecting singing/vocal event timestamps. A local FastAPI wrapper is available for testing; neither is connected to the ARTIST OS production site.

## Environment

The existing manifest validator remains lightweight. Install the optional runtime
only in a dedicated environment:

```bash
python -m pip install -r requirements-vocal.txt
```

Download the official FireRedVAD model files separately and point the detector at
the local directory containing the AED model files (including `model.pth.tar` and
`cmvn.ark`). The model is lazy-loaded on the first analysis request.

## API locale (prototype)

Installer les dépendances de l'API et du modèle dans un environnement dédié :

```bash
python -m pip install -r requirements-api.txt
```

Définir `FIREREDVAD_MODEL_DIR` vers le dossier `AED` du modèle téléchargé, puis démarrer l'API :

```bash
export FIREREDVAD_MODEL_DIR=/path/to/pretrained_models/FireRedVAD/AED
uvicorn app:app --host 127.0.0.1 --port 8000
```

L'endpoint `POST /analyze` attend un fichier multipart nommé `file` et renvoie `{"analysis": ...}`. `GET /health` vérifie la configuration sans charger les poids. Limite d'upload : 50 Mio.

Seul le WAV est accepté par l'API de test pour l'instant, car c'est le format réellement utilisé lors du test du modèle. FLAC, OGG, MP3 et M4A sont refusés jusqu'à validation du décodage par le modèle lui-même. L'API n'a ni authentification ni limitation de débit : ne pas l'exposer publiquement ni la connecter à ARTIST OS en production.

## Python use

```python
from src.vocal_detection import FireRedVADDetector

detector = FireRedVADDetector(
    model_dir="/path/to/pretrained_models/FireRedVAD/AED",
    use_gpu=False,
)
result = detector.analyze_file("/path/to/song.wav")
print(result["singing"]["segments"])
```

The JSON-compatible result contains duration and normalized timestamp segments for
`singing`, `speech`, and `music`. Ratios are descriptive model outputs, not
probabilities that a song contains vocals. Singing intervals must be checked against
listening before treating them as ground truth.

## Test de bout en bout avec le vrai modèle

Ce test charge les poids FireRedVAD et envoie un WAV réel à l'endpoint FastAPI via le client HTTP de test. Il mesure le temps total, vérifie le contrat JSON et affiche le nombre de segments détectés. Il ne téléverse aucun fichier vers un service distant.

Dans l'environnement où le modèle et le fichier audio existent déjà :

```bash
python -m pip install -r requirements-api.txt
python scripts/smoke_test_real_api.py \
  --audio /path/to/artist_os_test.wav \
  --model-dir /path/to/pretrained_models/FireRedVAD/AED
```

Ajouter `--gpu` uniquement si PyTorch/CUDA et le modèle sont configurés pour GPU. Le premier appel inclut le chargement du modèle ; consigner ce temps séparément d'une mesure à chaud si l'objectif est la latence répétée. Vérifier ensuite les segments en écoutant les passages correspondants : le test automatique ne juge pas la justesse musicale.

## Tests

The unit tests exercise interval sorting, overlap merging, duration clamping,
invalid values, and result normalization without importing FireRedVAD or downloading
weights:

```bash
python -m unittest discover -s tests -v
```

The inference test must be run separately on a locally available, rights-cleared
audio file and the exact model snapshot. Record model revision, Python/package
versions, CPU/GPU, cold-start time, inference time, and human review of detected
segments. Do not connect this prototype to production until repeatable tests and
license/dependency review are complete.
