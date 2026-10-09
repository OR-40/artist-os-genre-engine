import io
import tempfile
import unittest
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app


class FakeDetector:
    def analyze_file(self, path):
        return {
            "model": "FireRedVAD",
            "duration_seconds": 1.0,
            "singing": {"segments": [], "total_duration_seconds": 0.0, "ratio": 0.0},
        }


def make_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16000)
    return buffer.getvalue()


class VocalAnalysisApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app(detector=FakeDetector())
        self.app.state.model_dir = Path(self.temp_dir.name)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_health_does_not_load_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "FireRedVAD")
        self.assertEqual(response.json()["supported_formats"], [".wav"])

    def test_analyze_wav_returns_analysis_envelope(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("sample.wav", make_wav(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["analysis"]["model"], "FireRedVAD")
        self.assertEqual(response.json()["analysis"]["duration_seconds"], 1.0)

    def test_rejects_flac_until_real_model_decoder_is_validated(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("sample.flac", b"not-a-flac", "audio/flac")},
        )
        self.assertEqual(response.status_code, 415)

    def test_rejects_ogg_until_real_model_decoder_is_validated(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("sample.ogg", b"not-an-ogg", "audio/ogg")},
        )
        self.assertEqual(response.status_code, 415)

    def test_rejects_mp3_until_decoder_is_validated(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("sample.mp3", b"not-a-real-mp3", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 415)

    def test_rejects_corrupt_wav(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("sample.wav", b"not-a-wav", "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_empty_audio(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("empty.wav", b"", "audio/wav")},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
