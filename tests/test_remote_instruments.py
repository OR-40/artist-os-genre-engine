import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from src.remote_instruments import RemoteInstrumentAnalyzer


class RemoteInstrumentAnalyzerTests(unittest.TestCase):
    def test_disabled_when_url_is_absent_and_does_not_open_audio(self):
        analyzer = RemoteInstrumentAnalyzer(url="", timeout_seconds=1)
        self.assertFalse(analyzer.enabled)
        with patch("src.remote_instruments.httpx.post") as post:
            self.assertEqual(analyzer.analyze_file("/path/that/does/not/exist.wav"), [])
        post.assert_not_called()

    def test_calls_existing_service_contract_and_normalizes_predictions(self):
        analyzer = RemoteInstrumentAnalyzer(url="https://instruments.example", timeout_seconds=20)
        response = httpx.Response(
            200,
            request=httpx.Request("POST", "https://instruments.example/analyze"),
            json={
                "ok": True,
                "model": "onnx-community/Musical-Instrument-Classification-ONNX",
                "predictions": [
                    {"label": "Electric Guitar", "score": 0.72, "maxScore": 0.91, "occurrences": 5},
                    {"label": "Bass Guitar", "score": 0.61, "maxScore": 0.8, "occurrences": 3},
                    {"label": "", "score": 0.5, "maxScore": 0.5, "occurrences": 1},
                ],
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.wav"
            path.write_bytes(b"decoded wav bytes")
            with patch("src.remote_instruments.httpx.post", return_value=response) as post:
                result = analyzer.analyze_file(path)

        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://instruments.example/analyze")
        self.assertEqual(kwargs["files"]["audio"][0], "sample.wav")
        self.assertEqual(result[0]["name"], "Electric Guitar")
        self.assertEqual(result[0]["confidence"], 0.72)
        self.assertEqual(result[0]["max_score"], 0.91)
        self.assertEqual(result[0]["occurrences"], 5)
        self.assertEqual(result[1]["name"], "Bass Guitar")
        self.assertEqual(len(result), 2)

    def test_accepts_url_that_already_contains_analyze_path(self):
        analyzer = RemoteInstrumentAnalyzer(url="https://instruments.example/analyze", timeout_seconds=1)
        self.assertEqual(analyzer.url, "https://instruments.example/analyze")

    def test_rejects_unexpected_service_response(self):
        analyzer = RemoteInstrumentAnalyzer(url="https://instruments.example", timeout_seconds=1)
        response = httpx.Response(200, request=httpx.Request("POST", "https://instruments.example/analyze"), json={"ok": False, "error": "service unavailable"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.wav"
            path.write_bytes(b"decoded wav bytes")
            with patch("src.remote_instruments.httpx.post", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "réponse invalide"):
                    analyzer.analyze_file(path)


if __name__ == "__main__":
    unittest.main()
