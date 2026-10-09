import csv
import tempfile
import unittest
from pathlib import Path

from src.license_manifest import (
    FIELDS, eligible_rows, is_http_url, main, read_manifest, validate_rows, write_template
)


def valid_row(**overrides):
    row = {field: "" for field in FIELDS}
    row.update({
        "track_id": "track-001",
        "title": "Exemple",
        "source_url": "https://example.org/track",
        "audio_path": "data/audio/track-001.mp3",
        "license_name": "À vérifier",
        "license_url": "https://example.org/license",
        "training_use_permission": "unclear",
        "commercial_use_permission": "unclear",
        "rights_review_status": "pending",
        "genre_labels": "rock",
    })
    row.update(overrides)
    return row


class LicenseManifestTests(unittest.TestCase):
    def test_accepts_valid_pending_row(self):
        self.assertEqual(validate_rows([valid_row()]), [])

    def test_rejects_empty_manifest(self):
        self.assertTrue(any("sans piste" in error for error in validate_rows([])))

    def test_rejects_missing_required_field(self):
        errors = validate_rows([valid_row(license_url="")])
        self.assertTrue(any("license_url" in error for error in errors))

    def test_audio_path_is_required(self):
        errors = validate_rows([valid_row(audio_path="")])
        self.assertTrue(any("audio_path" in error for error in errors))

    def test_genre_labels_are_required(self):
        errors = validate_rows([valid_row(genre_labels="")])
        self.assertTrue(any("genre_labels" in error for error in errors))

    def test_rejects_duplicate_ids(self):
        self.assertTrue(any("dupliqué" in error for error in validate_rows([valid_row(), valid_row()])))

    def test_rejects_non_http_url(self):
        self.assertFalse(is_http_url("javascript:alert(1)"))
        self.assertFalse(is_http_url("https:///missing-host"))
        errors = validate_rows([valid_row(source_url="file:///tmp/song.mp3")])
        self.assertTrue(any("source_url" in error for error in errors))

    def test_accepts_http_and_https_urls(self):
        self.assertTrue(is_http_url("https://example.org/song"))
        self.assertTrue(is_http_url("http://example.org/song"))

    def test_approved_requires_reviewer_and_date(self):
        errors = validate_rows([valid_row(
            rights_review_status="approved",
            training_use_permission="yes",
            commercial_use_permission="yes",
            license_name="CC0-1.0",
        )])
        self.assertTrue(any("rights_reviewed_by" in error for error in errors))
        self.assertTrue(any("rights_review_date" in error for error in errors))

    def test_approved_requires_yes_for_both_uses(self):
        errors = validate_rows([valid_row(
            rights_review_status="approved",
            rights_reviewed_by="Reviewer",
            rights_review_date="2026-10-09",
            license_name="CC0-1.0",
        )])
        self.assertTrue(any("training_use_permission" in error for error in errors))
        self.assertTrue(any("commercial_use_permission" in error for error in errors))

    def test_approved_accepts_complete_human_review_record(self):
        row = valid_row(
            rights_review_status="approved",
            rights_reviewed_by="Reviewer",
            rights_review_date="2026-10-09",
            license_name="CC0-1.0",
            training_use_permission="yes",
            commercial_use_permission="yes",
        )
        self.assertEqual(validate_rows([row]), [])
        self.assertEqual(eligible_rows([row]), [row])

    def test_approved_rejects_unknown_license_name(self):
        errors = validate_rows([valid_row(
            rights_review_status="approved",
            rights_reviewed_by="Reviewer",
            rights_review_date="2026-10-09",
            license_name="UNKNOWN",
            training_use_permission="yes",
            commercial_use_permission="yes",
        )])
        self.assertTrue(any("license_name" in error for error in errors))

    def test_rejects_unknown_status(self):
        errors = validate_rows([valid_row(rights_review_status="probably-ok")])
        self.assertTrue(any("rights_review_status" in error for error in errors))

    def test_rejects_unknown_permission_value(self):
        errors = validate_rows([valid_row(training_use_permission="maybe")])
        self.assertTrue(any("training_use_permission" in error for error in errors))

    def test_non_string_cell_does_not_crash_validation(self):
        errors = validate_rows([valid_row(title=None)])
        self.assertTrue(any("title" in error for error in errors))

    def test_eligibility_filter_excludes_pending_rejected_and_unclear(self):
        approved = valid_row(
            track_id="approved",
            rights_review_status="approved",
            rights_reviewed_by="Reviewer",
            rights_review_date="2026-10-09",
            license_name="CC0-1.0",
            training_use_permission="yes",
            commercial_use_permission="yes",
        )
        pending = valid_row(track_id="pending")
        rejected = valid_row(track_id="rejected", rights_review_status="rejected")
        unclear = valid_row(
            track_id="unclear",
            rights_review_status="approved",
            rights_reviewed_by="Reviewer",
            rights_review_date="2026-10-09",
            license_name="CC0-1.0",
            training_use_permission="yes",
            commercial_use_permission="unclear",
        )
        self.assertEqual([row["track_id"] for row in eligible_rows([approved, pending, rejected, unclear])], ["approved"])

    def test_read_manifest_reports_missing_file(self):
        rows, errors = read_manifest(Path("file-that-should-not-exist.csv"))
        self.assertEqual(rows, [])
        self.assertTrue(any("introuvable" in error for error in errors))

    def test_read_manifest_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("track_id,title\n1,Test\n", encoding="utf-8")
            rows, errors = read_manifest(path)
        self.assertEqual(rows, [])
        self.assertTrue(any("colonnes manquantes" in error for error in errors))

    def test_read_manifest_rejects_duplicate_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate.csv"
            path.write_text("track_id,track_id\n1,2\n", encoding="utf-8")
            rows, errors = read_manifest(path)
        self.assertEqual(rows, [])
        self.assertTrue(any("dupliqués" in error for error in errors))

    def test_read_manifest_rejects_extra_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "extra.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(FIELDS)
                writer.writerow([valid_row()[field] for field in FIELDS] + ["unexpected"])
            rows, errors = read_manifest(path)
        self.assertEqual(rows, [])
        self.assertTrue(any("trop de valeurs" in error for error in errors))

    def test_template_writes_expected_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "template.csv"
            write_template(path)
            with path.open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle))
        self.assertEqual(header, FIELDS)

    def test_eligible_export_writes_only_approved_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "manifest.csv"
            output = Path(tmp) / "eligible.csv"
            rows = [
                valid_row(
                    track_id="approved",
                    rights_review_status="approved",
                    rights_reviewed_by="Reviewer",
                    rights_review_date="2026-10-09",
                    license_name="CC0-1.0",
                    training_use_permission="yes",
                    commercial_use_permission="yes",
                ),
                valid_row(track_id="pending"),
            ]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            result = main(["eligible", str(source), "--output", str(output)])
            with output.open(encoding="utf-8", newline="") as handle:
                exported = list(csv.DictReader(handle))
        self.assertEqual(result, 0)
        self.assertEqual([row["track_id"] for row in exported], ["approved"])

    def test_eligible_export_refuses_to_overwrite_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "manifest.csv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(valid_row())
            result = main(["eligible", str(source), "--output", str(source)])
            self.assertEqual(result, 2)
            self.assertTrue(source.exists())


    def test_real_music_pilot_candidates_manifest_is_structurally_valid_and_pending(self):
        path = Path("data/pilot_candidates.csv")
        rows, read_errors = read_manifest(path)
        self.assertEqual(read_errors, [])
        self.assertEqual(validate_rows(rows), [])
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row["rights_review_status"] == "pending" for row in rows))
        self.assertTrue(all(row["training_use_permission"] == "unclear" for row in rows))
        self.assertTrue(all(row["commercial_use_permission"] == "unclear" for row in rows))
        self.assertEqual(len({row["track_id"] for row in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()
