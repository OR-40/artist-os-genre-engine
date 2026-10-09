import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import soundfile as sf

from src.artist_dna import ArtistDNAEngine, DEFAULT_GENRE_MODELS
from src.instrument_classifier import LocalONNXInstrumentClassifier, _sample_starts


class FakeClassifier:
    def __init__(self, name, model_id, labels):
        self.name = name
        self.model_id = model_id
        self.labels = labels
        self.calls = 0

    def predict(self, audio, sample_rate, top_k=5):
        self.calls += 1
        return [{"label": label, "score": score} for label, score in self.labels]


class WindowSensitiveClassifier:
    name = "window_sensitive"
    model_id = "fake/window-sensitive"

    def __init__(self):
        self.calls = 0

    def predict(self, audio, sample_rate, top_k=10):
        self.calls += 1
        if self.calls == 2:
            return [{"label": "classical", "score": 0.95}, {"label": "rock", "score": 0.03}]
        return [{"label": "rock", "score": 0.72}, {"label": "classical", "score": 0.12}]


class FakeInstrumentAnalyzer:
    enabled = True
    model_id = "fake/instruments"

    def analyze(self, audio, sample_rate, top_k=8):
        return [{"name": "Electric Guitar", "confidence": 0.7, "top1_windows": 2, "windows_analyzed": 3, "role": ""}]


class LowConfidenceInstrumentAnalyzer:
    enabled = True
    model_id = "fake/single-instrument-model"

    def analyze_file(self, path):
        return [{
            "name": "Violin",
            "confidence": 0.2082,
            "top1_windows": 4,
            "windows_analyzed": 6,
            "role": "",
        }]


class FailingInstrumentAnalyzer:
    enabled = True
    model_id = "fake/remote-instruments"

    def analyze_file(self, path):
        raise TimeoutError("instrument service timeout")


class FakeVocalDetector:
    def analyze_file(self, path):
        return {
            "model": "FireRedVAD",
            "duration_seconds": 12.0,
            "singing": {"segments": [{"start": 2.0, "end": 5.0, "duration": 3.0}],
                        "total_duration_seconds": 3.0, "ratio": 0.25},
            "speech": {"segments": [], "total_duration_seconds": 0.0, "ratio": 0.0},
            "music": {"segments": [], "total_duration_seconds": 0.0, "ratio": 0.0},
        }


class ArtistDNAEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.tmp.name) / "sample.wav"
        sf.write(self.audio_path, np.zeros(12 * 16000, dtype=np.float32), 16000)
        self.a = FakeClassifier("candidate_a", "fake/a", [("rock", 0.8), ("metal", 0.2)])
        self.b = FakeClassifier("candidate_b", "fake/b", [("rock", 0.7), ("pop", 0.3)])
        self.engine = ArtistDNAEngine(
            FakeVocalDetector(),
            [self.a, self.b],
            window_seconds=30,
            instrument_analyzer=FakeInstrumentAnalyzer(),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_pipeline_configures_baseline_and_loadable_ast_checkpoint(self):
        with patch.dict("os.environ", {}, clear=True):
            engine = ArtistDNAEngine(FakeVocalDetector())
        self.assertEqual(
            [item[0] for item in DEFAULT_GENRE_MODELS],
            ["genre_baseline", "genre_ast"],
        )
        self.assertEqual(
            [item.name for item in engine.classifiers],
            ["genre_baseline", "genre_ast"],
        )
        self.assertEqual(engine.classifiers[0].window_seconds, 30)
        self.assertEqual(engine.classifiers[1].window_seconds, 10)
        self.assertEqual(engine.classifiers[1].max_windows, 24)
        self.assertEqual(
            engine.classifiers[1].model_id,
            "Koras1k/ast-megafinetuned-gtzan-v2-0.97score",
        )

    def test_combines_two_genre_candidates_and_vocal_detection(self):
        result = self.engine.analyze_file(self.audio_path)
        self.assertEqual(result["engine"], "ARTIST DNA")
        self.assertEqual(result["engine_version"], "0.2.0")
        self.assertEqual(set(result["genre_analysis"]["models"]), {"candidate_a", "candidate_b"})
        self.assertEqual(result["vocal_analysis"]["model"], "FireRedVAD")
        self.assertEqual(result["vocal_analysis"]["singing"]["total_duration_seconds"], 3.0)
        self.assertEqual(result["genre_analysis"]["label_consensus"][0]["label"], "rock")
        self.assertEqual(result["genre_analysis"]["label_consensus"][0]["models_agreeing"], 2)
        self.assertEqual(result["genre_analysis"]["label_consensus"][0]["model_count"], 2)
        self.assertEqual(self.a.calls, 1)
        self.assertEqual(self.b.calls, 1)

    def test_long_song_uses_full_track_coverage_in_30_second_windows(self):
        long_path = Path(self.tmp.name) / "long.wav"
        sf.write(long_path, np.zeros(120 * 16000, dtype=np.float32), 16000)
        result = self.engine.analyze_file(long_path)
        self.assertEqual(result["analysis_sampling"]["selected_audio_seconds"], 120.0)
        self.assertEqual(len(result["analysis_sampling"]["selected_windows"]), 4)
        self.assertEqual([w["start_seconds"] for w in result["analysis_sampling"]["selected_windows"]], [0.0, 30.0, 60.0, 90.0])
        self.assertEqual(result["genre_analysis"]["decision_status"], "repeated_temporal_evidence")
        self.assertEqual(result["genre_analysis"]["window_top1_votes"][0]["label"], "rock")

    def test_single_outlier_window_does_not_define_whole_track_genre(self):
        long_path = Path(self.tmp.name) / "long-outlier.wav"
        sf.write(long_path, np.zeros(120 * 16000, dtype=np.float32), 16000)
        classifier = WindowSensitiveClassifier()
        engine = ArtistDNAEngine(
            FakeVocalDetector(),
            [classifier],
            window_seconds=30,
            instrument_analyzer=LocalONNXInstrumentClassifier(enabled=False),
        )
        result = engine.analyze_file(long_path)
        self.assertEqual(result["genres"][0], "rock")
        self.assertNotIn("classical", result["genres"])
        self.assertEqual(result["genre_analysis"]["decision_status"], "repeated_temporal_evidence")
        self.assertEqual(result["artistic_analysis"]["status"], "provisional_evidence_based")
        self.assertIn("rock", result["artistic_analysis"]["text"])

    def test_marks_signature_as_not_generated_instead_of_inventing(self):
        result = self.engine.analyze_file(self.audio_path)
        self.assertEqual(result["artistic_signature"]["status"], "not_generated")

    def test_instrument_candidate_is_marked_experimental(self):
        result = self.engine.analyze_file(self.audio_path)
        self.assertEqual(result["instrument_analysis"]["status"], "experimental_mix_generalization_unvalidated")
        self.assertEqual(result["instrument_analysis"]["predictions"][0]["name"], "Electric Guitar")

    def test_instrument_service_failure_is_fail_closed(self):
        engine = ArtistDNAEngine(
            FakeVocalDetector(),
            [self.a, self.b],
            window_seconds=10,
            instrument_analyzer=FailingInstrumentAnalyzer(),
        )
        result = engine.analyze_file(self.audio_path)
        self.assertEqual(result["instrument_analysis"]["status"], "unavailable")
        self.assertEqual(result["instrument_analysis"]["predictions"], [])
        self.assertEqual(result["genre_analysis"]["label_consensus"][0]["label"], "rock")
        self.assertEqual(result["genres"][0], "rock")
        self.assertEqual(result["genres"], ["rock"])
        self.assertIn("genre vocal ne sont pas évalués", result["voice"])
        self.assertEqual(result["instrumentation"], [])

    def test_catalog_compatibility_fields_include_actual_instrument_predictions(self):
        result = self.engine.analyze_file(self.audio_path)
        self.assertEqual(result["genres"][0], "rock")
        self.assertEqual(result["instrumentation"][0]["name"], "Electric Guitar")
        self.assertIn("genre vocal ne sont pas évalués", result["voice"])

    def test_weak_violin_candidate_is_not_promoted_to_confirmed_instrumentation(self):
        engine = ArtistDNAEngine(
            FakeVocalDetector(),
            [self.a, self.b],
            window_seconds=30,
            instrument_analyzer=LowConfidenceInstrumentAnalyzer(),
        )
        result = engine.analyze_file(self.audio_path)
        self.assertEqual(result["instrumentation"], [])
        self.assertEqual(result["instrument_analysis"]["predictions"][0]["name"], "Violin")
        self.assertEqual(result["instrument_analysis"]["confirmed_predictions"], [])

    def test_disabled_instrument_candidate_does_not_run_model(self):
        engine = ArtistDNAEngine(
            FakeVocalDetector(),
            [self.a, self.b],
            window_seconds=10,
            instrument_analyzer=LocalONNXInstrumentClassifier(enabled=False),
        )
        result = engine.analyze_file(self.audio_path)
        self.assertEqual(result["instrument_analysis"]["status"], "not_enabled")
        self.assertEqual(result["instrument_analysis"]["predictions"], [])

    def test_local_onnx_is_enabled_by_default_without_remote_service(self):
        with patch.dict("os.environ", {}, clear=True):
            analyzer = LocalONNXInstrumentClassifier()
        self.assertTrue(analyzer.enabled)
        self.assertEqual(analyzer.model_id, "onnx-community/Musical-Instrument-Classification-ONNX")

    def test_sample_windows_are_evenly_spaced_and_bounded(self):
        self.assertEqual(_sample_starts(16000 * 10, 16000 * 3, 4), [0, 37333, 74667, 112000])
        self.assertEqual(_sample_starts(100, 300, 6), [0])

    def test_local_onnx_adapter_uses_injected_session_without_downloading(self):
        class Input:
            name = "input_values"

        class FakeSession:
            def get_inputs(self):
                return [Input()]

            def run(self, outputs, feeds):
                self.assert_input = feeds["input_values"]
                return [np.array([[0.0, 2.0]], dtype=np.float32)]

        class FakeFeatureExtractor:
            def __call__(self, audio, sampling_rate, return_tensors):
                self.assert_rate = sampling_rate
                self.assert_shape = np.asarray(audio).shape
                return {"input_values": np.asarray(audio, dtype=np.float32)[None, :]}

        session = FakeSession()
        adapter = LocalONNXInstrumentClassifier(
            enabled=True,
            session=session,
            feature_extractor=FakeFeatureExtractor(),
            labels={0: "Acoustic_Guitar", 1: "Bass_Guitar"},
            max_segments=1,
        )
        result = adapter.analyze(np.zeros(16000 * 3, dtype=np.float32), 16000, top_k=2)
        self.assertEqual(result[0]["name"], "Bass Guitar")
        self.assertEqual(result[0]["windows_analyzed"], 1)
        self.assertEqual(session.assert_input.shape, (1, 48000))

    def test_missing_file_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            self.engine.analyze_file(Path(self.tmp.name) / "missing.wav")


if __name__ == "__main__":
    unittest.main()
