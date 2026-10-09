import math
import unittest

from src.vocal_detection import normalize_result, normalize_segments, summarize_event


class VocalDetectionTests(unittest.TestCase):
    def test_sorts_and_merges_overlapping_intervals(self):
        segments = normalize_segments([(4, 6), (1, 3), (2.5, 4.5)], 10)
        self.assertEqual([(s.start, s.end) for s in segments], [(1.0, 6.0)])

    def test_clamps_intervals_to_audio_duration(self):
        segments = normalize_segments([(-2, 2), (8, 14)], 10)
        self.assertEqual([(s.start, s.end) for s in segments], [(0.0, 2.0), (8.0, 10.0)])

    def test_discards_empty_or_outside_intervals(self):
        segments = normalize_segments([(3, 3), (11, 12), (-4, -1)], 10)
        self.assertEqual(segments, [])

    def test_rejects_bad_intervals(self):
        with self.assertRaises(ValueError):
            normalize_segments([(1,)], 10)
        with self.assertRaises(ValueError):
            normalize_segments([("x", 2)], 10)
        with self.assertRaises(ValueError):
            normalize_segments([(0, math.inf)], 10)

    def test_rejects_nonpositive_duration(self):
        with self.assertRaises(ValueError):
            normalize_segments([], 0)

    def test_summarizes_mapping_result(self):
        result = {
            "event2timestamps": {
                "singing": [[1, 3], [4, 6]],
                "speech": [],
                "music": [[0, 10]],
            }
        }
        summary = normalize_result(result, 10)
        self.assertEqual(summary["model"], "FireRedVAD")
        self.assertEqual(summary["singing"]["total_duration_seconds"], 4.0)
        self.assertEqual(summary["singing"]["ratio"], 0.4)
        self.assertEqual(summary["speech"]["segments"], [])
        self.assertEqual(summary["music"]["ratio"], 1.0)

    def test_supports_result_object(self):
        class Result:
            event2timestamps = {"singing": [[2, 5]]}

        summary = summarize_event(Result(), "singing", 10)
        self.assertEqual(summary["total_duration_seconds"], 3.0)
        self.assertEqual(summary["ratio"], 0.3)

    def test_rejects_invalid_event_map(self):
        with self.assertRaises(ValueError):
            normalize_result({"event2timestamps": None}, 10)


if __name__ == "__main__":
    unittest.main()
