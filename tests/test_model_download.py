import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.vocal_detection import FireRedVADDetector, REQUIRED_AED_FILES


class FireRedVADDownloadTests(unittest.TestCase):
    def test_downloads_only_missing_official_aed_files_into_model_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "weights" / "FireRedVAD" / "AED"
            calls = []

            def fake_download(repo_id, filename, local_dir):
                calls.append((repo_id, filename, local_dir))
                destination = Path(local_dir) / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"test model file")
                return str(destination)

            fake_hub = SimpleNamespace(hf_hub_download=fake_download)
            detector = FireRedVADDetector(model_dir=model_dir)

            with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
                detector._ensure_model_files()

            self.assertEqual([c[1] for c in calls], [f"AED/{name}" for name in REQUIRED_AED_FILES])
            self.assertTrue(all((model_dir / name).is_file() for name in REQUIRED_AED_FILES))
            self.assertTrue(all(c[0] == "FireRedTeam/FireRedVAD" for c in calls))
            self.assertTrue(all(c[2] == str(model_dir.parent) for c in calls))

    def test_does_not_download_when_both_model_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "AED"
            model_dir.mkdir()
            for name in REQUIRED_AED_FILES:
                (model_dir / name).write_bytes(b"already present")

            detector = FireRedVADDetector(model_dir=model_dir)
            fake_hub = SimpleNamespace(
                hf_hub_download=lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected download"))
            )
            with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
                detector._ensure_model_files()

    def test_raises_clear_error_if_download_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "AED"
            detector = FireRedVADDetector(model_dir=model_dir)
            fake_hub = SimpleNamespace(
                hf_hub_download=lambda **kwargs: (_ for _ in ()).throw(OSError("offline"))
            )
            with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
                with self.assertRaisesRegex(RuntimeError, "Téléchargement des poids officiels"):
                    detector._ensure_model_files()


if __name__ == "__main__":
    unittest.main()
