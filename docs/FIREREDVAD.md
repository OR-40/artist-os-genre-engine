# FireRedVAD — ARTIST DNA prototype

This module is an isolated prototype for detecting singing, speech and music events. The FastAPI wrapper and the combined ARTIST DNA engine are not connected to the production ARTIST OS site.

## One-time environment setup

Run in the repository root:

```bash
python -m pip install -r requirements-dna.txt
```

This installs the API, FireRedVAD and genre-classification dependencies. The API requires FFmpeg to decode MP3 files to temporary mono 16 kHz PCM WAV files.

## Model weights

On the first real inference, `FireRedVADDetector` downloads the two required AED files (`AED/model.pth.tar` and `AED/cmvn.ark`) from the official `FireRedTeam/FireRedVAD` Hugging Face repository if they are not already present. The AED checkpoint is approximately 2.37 MB; the CMVN file is approximately 1.31 kB. Genre models are loaded from Hugging Face by Transformers on the first full ARTIST DNA request. Instrument classification is disabled by default because it is experimental on mixed songs.

Optional environment settings:

```bash
export FIREREDVAD_MODEL_DIR=weights/FireRedVAD/AED
export FIREREDVAD_USE_GPU=false
```

## Real MP3 smoke test

Place a rights-cleared MP3 in the repository or pass its absolute path. The test exercises the actual API and reports elapsed time.

FireRedVAD only:

```bash
python scripts/smoke_test_real_api.py --audio "/path/to/song.mp3"
```

Full ARTIST DNA (two genre candidates plus FireRedVAD):

```bash
python scripts/smoke_test_real_api.py --audio "/path/to/song.mp3" --dna
```

The first full ARTIST DNA request downloads and loads the genre models, so its cold-start time includes those downloads. Subsequent requests in the same process reuse the loaded models. The current genre candidates are experimental; their scores are model outputs, not calibrated probabilities, and the artistic signature is intentionally not generated from genre scores alone.

## API contract

- `POST /analyze`: FireRedVAD vocal-event analysis.
- `POST /dna/analyze`: combined ARTIST DNA analysis.
- Both endpoints accept an MP3 multipart field named `file`, enforce a 50 MiB upload limit and return `{"analysis": ...}`.
- `GET /health` checks configuration without loading model weights.
- The WAV produced from MP3 is temporary and is deleted after the request.
- The prototype has no authentication or rate limiting. Do not expose it publicly or connect it to production.

## Unit tests

```bash
python -m unittest discover -s tests -v
```

Unit tests use fake models for contract logic and do not validate real model quality. Real inference must be checked separately, including listening to detected vocal segments and reviewing genre predictions before using them as evidence about an artist.
