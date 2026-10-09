import unittest
from src.license_manifest import FIELDS, validate_rows, is_http_url


def valid_row(**overrides):
    row = {field: "" for field in FIELDS}
    row.update({
        "track_id": "track-001",
        "title": "Exemple",
        "source_url": "https://example.org/track",
        "license_name": "À vérifier",
        "license_url": "https://example.org/license",
        "rights_review_status": "pending",
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

    def test_rejects_duplicate_ids(self):
        errors = validate_rows([valid_row(), valid_row()])
        self.assertTrue(any("dupliqué" in error for error in errors))

    def test_rejects_non_http_url(self):
        self.assertFalse(is_http_url("javascript:alert(1)"))
        errors = validate_rows([valid_row(source_url="file:///tmp/song.mp3")])
        self.assertTrue(any("source_url" in error for error in errors))

    def test_approved_requires_reviewer_and_date(self):
        errors = validate_rows([valid_row(rights_review_status="approved")])
        self.assertTrue(any("rights_reviewed_by" in error for error in errors))
        self.assertTrue(any("rights_review_date" in error for error in errors))

    def test_rejects_unknown_status(self):
        errors = validate_rows([valid_row(rights_review_status="probably-ok")])
        self.assertTrue(any("rights_review_status" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
