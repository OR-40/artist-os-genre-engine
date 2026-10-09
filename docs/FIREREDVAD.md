# FireRedVAD — vocal-event detection prototype

This module is an isolated prototype for detecting singing/vocal event timestamps.
It is not yet connected to a web API or to the ARTIST OS production site.

## Environment

The existing manifest validator remains lightweight. Install the optional runtime
only in a dedicated environment:

```bash
python -m pip install -r requirements-vocal.txt
```

Download the official FireRedVAD model files separately and point the detector at
the local directory containing the AED model files (including `model.pth.tar` and
`cmvn.ark`). The model is lazy-loaded on the first analysis request.

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
