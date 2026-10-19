"""Regression tests for the deployment smoke-test helper."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_test_deployment.py"
SPEC = importlib.util.spec_from_file_location("artist_dna_smoke_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class DeploymentSmokeScriptTests(unittest.TestCase):
    def test_script_compiles_and_imports_without_network_access(self) -> None:
        self.assertTrue(callable(SMOKE.main))

    def test_multipart_audio_uses_expected_audio_field_and_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "test-track.mp3"
            audio.write_bytes(b"fake-mp3-payload")
            boundary, body = SMOKE.multipart_audio(audio)

        self.assertTrue(boundary.startswith("----ArtistDNA"))
        self.assertIn(b'name="audio"; filename="test-track.mp3"', body)
        self.assertIn(b"fake-mp3-payload", body)
        self.assertTrue(body.endswith(f"--{boundary}--\r\n".encode()))

    def test_multipart_audio_does_not_embed_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "track.mp3"
            audio.write_bytes(b"payload")
            _, body = SMOKE.multipart_audio(audio)

        self.assertNotIn(b"Authorization", body)
        self.assertNotIn(b"Bearer ", body)


if __name__ == "__main__":
    unittest.main()
