import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import create_app


class FakeDNAEngine:
    def analyze_file(self, path):
        assert Path(path).suffix == ".wav"
        return {"engine": "ARTIST DNA", "genre_analysis": {"models": {}}, "vocal_analysis": {}}


class FakeDetector:
    def analyze_file(self, path):
        assert Path(path).suffix == ".wav"
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


def fake_decode_mp3_to_wav(source_path, destination_path):
    Path(destination_path).write_bytes(make_wav())


class VocalAnalysisApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = create_app(detector=FakeDetector())
        self.app.state.model_dir = Path(self.temp_dir.name)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_health_does_not_load_model_and_advertises_mp3(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "FireRedVAD")
        self.assertEqual(response.json()["supported_formats"], [".mp3"])

    def test_analyze_mp3_returns_analysis_envelope(self):
        with patch("app.decode_mp3_to_wav", side_effect=fake_decode_mp3_to_wav):
            response = self.client.post(
                "/analyze",
                files={"file": ("sample.mp3", b"mocked MP3 payload", "audio/mpeg")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["analysis"]["model"], "FireRedVAD")
        self.assertEqual(response.json()["analysis"]["duration_seconds"], 1.0)

    def test_dna_endpoint_returns_combined_analysis_contract_for_mp3(self):
        app = create_app(detector=FakeDetector(), dna_engine=FakeDNAEngine())
        client = TestClient(app)
        try:
            with patch("app.decode_mp3_to_wav", side_effect=fake_decode_mp3_to_wav):
                response = client.post(
                    "/dna/analyze",
                    files={"file": ("sample.mp3", b"mocked MP3 payload", "audio/mpeg")},
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["analysis"]["engine"], "ARTIST DNA")
        finally:
            client.close()

    def test_rejects_wav_because_upload_contract_is_mp3_only(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("sample.wav", make_wav(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 415)

    def test_rejects_corrupt_mp3(self):
        with patch("app.decode_mp3_to_wav", side_effect=ValueError("bad mp3")):
            response = self.client.post(
                "/analyze",
                files={"file": ("sample.mp3", b"not-a-real-mp3", "audio/mpeg")},
            )
        self.assertEqual(response.status_code, 400)

    def test_rejects_empty_audio(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("empty.mp3", b"", "audio/mpeg")},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
