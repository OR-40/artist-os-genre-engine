import unittest

from src.artistic_report import build_artistic_report


class ArtisticReportTests(unittest.TestCase):
    def test_report_only_claims_evidence_provided_to_it(self):
        report = build_artistic_report(
            genres=["rock"],
            genre_decision_status="repeated_temporal_evidence",
            singing={"total_duration_seconds": 69.5},
            confirmed_instruments=[],
            duration_seconds=224.544,
        )
        self.assertEqual(report["status"], "provisional_evidence_based")
        self.assertIn("rock", report["text"])
        self.assertIn("69.5 secondes", report["text"])
        self.assertIn("n'est pas validée", report["text"])
        self.assertIn("sous-genre", report["text"])

    def test_unstable_genre_is_reported_as_unstable(self):
        report = build_artistic_report(
            genres=["classical"],
            genre_decision_status="low_temporal_evidence",
            singing={"total_duration_seconds": 0},
            confirmed_instruments=[],
            duration_seconds=60,
        )
        self.assertEqual(report["status"], "low_confidence")
        self.assertIn("ne se répète pas assez", report["text"])
        self.assertIn("ne prouve pas", report["text"])


if __name__ == "__main__":
    unittest.main()
