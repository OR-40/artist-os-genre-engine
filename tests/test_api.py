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


class ExplodingDNAEngine:
    def analyze_file(self, path):
        raise TypeError("diagnostic test failure")


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
        self.app = create_app(detector=FakeDetector(), api_token='test-api-token')
        self.app.state.model_dir = Path(self.temp_dir.name)
        self.client = TestClient(self.app, headers={'Authorization': 'Bearer test-api-token'})

    def tearDown(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_health_does_not_load_model_and_advertises_mp3(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "FireRedVAD")
        self.assertEqual(response.json()["execution_device"], "CPU")
        self.assertEqual(response.json()["supported_formats"], [".flac", ".m4a", ".mp3", ".ogg", ".wav"])

    def test_analyze_mp3_returns_analysis_envelope(self):
        with patch("app.decode_audio_to_wav", side_effect=fake_decode_mp3_to_wav):
            response = self.client.post(
                "/analyze",
                files={"file": ("sample.mp3", b"mocked MP3 payload", "audio/mpeg")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["analysis"]["model"], "FireRedVAD")
        self.assertEqual(response.json()["analysis"]["duration_seconds"], 1.0)

    def test_dna_endpoint_returns_combined_analysis_contract_for_mp3(self):
        app = create_app(detector=FakeDetector(), dna_engine=FakeDNAEngine(), api_token='test-api-token')
        client = TestClient(app, headers={'Authorization': 'Bearer test-api-token'})
        try:
            with patch("app.decode_audio_to_wav", side_effect=fake_decode_mp3_to_wav):
                response = client.post(
                    "/dna/analyze",
                    files={"audio": ("sample.mp3", b"mocked MP3 payload", "audio/mpeg")},
                )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["ok"])
            self.assertEqual(response.json()["dna"]["engine"], "ARTIST DNA")
        finally:
            client.close()

    def test_dna_endpoint_accepts_all_advertised_audio_formats(self):
        app = create_app(detector=FakeDetector(), dna_engine=FakeDNAEngine(), api_token='test-api-token')
        client = TestClient(app, headers={'Authorization': 'Bearer test-api-token'})
        samples = [
            ("sample.mp3", "audio/mpeg"),
            ("sample.wav", "audio/wav"),
            ("sample.flac", "audio/flac"),
            ("sample.ogg", "audio/ogg"),
            ("sample.m4a", "audio/mp4"),
        ]
        try:
            with patch("app.decode_audio_to_wav", side_effect=fake_decode_mp3_to_wav):
                for filename, content_type in samples:
                    with self.subTest(filename=filename):
                        response = client.post(
                            "/dna/analyze",
                            files={"audio": (filename, b"mocked audio payload", content_type)},
                        )
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertTrue(response.json()["ok"])
        finally:
            client.close()

    def test_dna_endpoint_rejects_missing_audio(self):
        app = create_app(detector=FakeDetector(), dna_engine=FakeDNAEngine(), api_token='test-api-token')
        client = TestClient(app, headers={'Authorization': 'Bearer test-api-token'})
        try:
            response = client.post("/dna/analyze")
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["detail"], "Fichier audio manquant.")
        finally:
            client.close()

    def test_dna_inference_failure_is_logged_and_returns_generic_502(self):
        app = create_app(detector=FakeDetector(), dna_engine=ExplodingDNAEngine(), api_token='test-api-token')
        client = TestClient(app, headers={'Authorization': 'Bearer test-api-token'})
        try:
            with patch("app.decode_audio_to_wav", side_effect=fake_decode_mp3_to_wav):
                with self.assertLogs("artist_os_genre_engine", level="ERROR") as captured:
                    response = client.post(
                        "/dna/analyze",
                        files={"file": ("sample.mp3", b"mocked MP3 payload", "audio/mpeg")},
                    )
            self.assertEqual(response.status_code, 502)
            self.assertIn("diagnostic test failure", "\\n".join(captured.output))
            self.assertIn("Échec d'un composant d'analyse ARTIST DNA", response.text)
        finally:
            client.close()

    def test_rejects_unsupported_audio_format(self):
        response = self.client.post(
            "/analyze",
            files={"file": ("sample.aac", b"unsupported", "audio/aac")},
        )
        self.assertEqual(response.status_code, 415)

    def test_rejects_corrupt_mp3(self):
        with patch("app.decode_audio_to_wav", side_effect=ValueError("bad mp3")):
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


    def test_inference_routes_require_bearer_token(self):
        app = create_app(detector=FakeDetector(), dna_engine=FakeDNAEngine(), api_token='test-api-token')
        client = TestClient(app)
        try:
            response = client.post(
                '/dna/analyze',
                files={'audio': ('sample.mp3', b'mocked audio payload', 'audio/mpeg')},
            )
            self.assertEqual(response.status_code, 401)
            self.assertIn('Authentification', response.json()['detail'])
        finally:
            client.close()

    def test_inference_fails_closed_when_server_token_is_missing(self):
        app = create_app(detector=FakeDetector(), dna_engine=FakeDNAEngine(), api_token='')
        client = TestClient(app, headers={'Authorization': 'Bearer anything'})
        try:
            response = client.post(
                '/dna/analyze',
                files={'audio': ('sample.mp3', b'mocked audio payload', 'audio/mpeg')},
            )
            self.assertEqual(response.status_code, 503)
            self.assertIn('non configuré', response.json()['detail'])
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
