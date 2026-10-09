"""Validation conservatrice d'un manifeste de provenance/licence audio.

Ce module ne télécharge pas d'audio et ne décide pas si une licence autorise
juridiquement un usage donné. La validation des droits reste humaine.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from urllib.parse import urlparse

FIELDS = [
    "track_id",
    "title",
    "artist",
    "source_url",
    "audio_path",
    "license_name",
    "license_url",
    "rights_review_status",
    "rights_reviewed_by",
    "rights_review_date",
    "genre_labels",
    "notes",
]
STATUSES = {"pending", "approved", "rejected"}
URL_FIELDS = ("source_url", "license_url")
REQUIRED_FIELDS = (
    "track_id",
    "title",
    "source_url",
    "audio_path",
    "license_name",
    "license_url",
    "rights_review_status",
    "genre_labels",
)


def _text(value: object) -> str:
    """Convertit une cellule CSV absente ou non textuelle en chaîne sûre."""
    return value.strip() if isinstance(value, str) else ""


def is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value.strip())
    except (AttributeError, ValueError):
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_rows(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    if not rows:
        return ["manifeste sans piste: ajoute au moins une ligne de données"]

    for row_number, row in enumerate(rows, start=2):
        for field in REQUIRED_FIELDS:
            if not _text(row.get(field)):
                errors.append(f"ligne {row_number}: champ obligatoire vide: {field}")

        track_id = _text(row.get("track_id"))
        if track_id:
            if track_id in seen_ids:
                errors.append(f"ligne {row_number}: track_id dupliqué: {track_id}")
            seen_ids.add(track_id)

        for field in URL_FIELDS:
            value = _text(row.get(field))
            if value and not is_http_url(value):
                errors.append(f"ligne {row_number}: URL HTTP(S) invalide dans {field}: {value}")

        status = _text(row.get("rights_review_status")).lower()
        if status and status not in STATUSES:
            errors.append(
                f"ligne {row_number}: rights_review_status doit être l'un de {', '.join(sorted(STATUSES))}"
            )

        if status == "approved":
            if _text(row.get("license_name")).upper() in {"UNKNOWN", "À VÉRIFIER", "A VERIFIER"}:
                errors.append(
                    f"ligne {row_number}: license_name doit identifier la licence lorsque rights_review_status=approved"
                )
            for field in ("rights_reviewed_by", "rights_review_date"):
                if not _text(row.get(field)):
                    errors.append(
                        f"ligne {row_number}: {field} obligatoire lorsque rights_review_status=approved"
                    )

    return errors


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], [f"fichier introuvable: {path}"]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return [], ["CSV vide ou sans en-tête"]
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                return [], ["en-tête CSV invalide: noms de colonnes dupliqués"]
            missing = [field for field in FIELDS if field not in reader.fieldnames]
            if missing:
                return [], ["colonnes manquantes: " + ", ".join(missing)]

            rows: list[dict[str, str]] = []
            errors: list[str] = []
            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    errors.append(f"ligne {row_number}: trop de valeurs par rapport aux colonnes")
                    continue
                rows.append(row)
            if errors:
                return [], errors
            return rows, []
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], [f"lecture CSV impossible: {exc}"]


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation du manifeste audio ARTIST OS")
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser("template", help="créer un CSV avec l'en-tête attendu")
    template_parser.add_argument("path", type=Path)

    validate_parser = subparsers.add_parser("validate", help="valider un manifeste CSV")
    validate_parser.add_argument("path", type=Path)

    args = parser.parse_args(argv)
    if args.command == "template":
        try:
            write_template(args.path)
        except OSError as exc:
            print(f"ERREUR: création impossible: {exc}", file=sys.stderr)
            return 2
        print(f"Modèle créé: {args.path}")
        return 0

    rows, read_errors = read_manifest(args.path)
    errors = read_errors + (validate_rows(rows) if not read_errors else [])
    if errors:
        print("MANIFESTE INVALIDE")
        for error in errors:
            print(f"- {error}")
        return 1

    approved = sum(_text(row.get("rights_review_status")).lower() == "approved" for row in rows)
    pending = sum(_text(row.get("rights_review_status")).lower() == "pending" for row in rows)
    rejected = sum(_text(row.get("rights_review_status")).lower() == "rejected" for row in rows)
    print(f"MANIFESTE VALIDE — {len(rows)} piste(s)")
    print(f"Droits vérifiés humainement (candidates) : {approved}")
    print(f"En attente de vérification : {pending}")
    print(f"Exclues : {rejected}")
    print("Attention : valide la structure du manifeste, pas la légalité des pistes ni leur admissibilité à l'entraînement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
