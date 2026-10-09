"""Evidence-grounded French artistic report for ARTIST DNA.

This module deliberately describes only signals that the current pipeline can
support. It does not invent a subgenre, vocal timbre, or instrument role.
"""
from __future__ import annotations

from typing import Any


def build_artistic_report(
    genres: list[str],
    genre_decision_status: str,
    singing: dict[str, Any],
    confirmed_instruments: list[dict[str, Any]],
    duration_seconds: float,
) -> dict[str, str]:
    """Create a concise French report from structured evidence."""
    duration = max(0, int(round(float(duration_seconds))))
    minutes, seconds = divmod(duration, 60)
    duration_label = f"{minutes} min {seconds:02d} s"

    if genre_decision_status == "repeated_temporal_evidence" and genres:
        primary = genres[0].replace("_", " ").strip().casefold()
        genre_sentence = f"Le style dominant estimé par les fenêtres réparties sur le morceau est « {primary} »."
        status = "provisional_evidence_based"
    elif genres:
        primary = genres[0].replace("_", " ").strip().casefold()
        genre_sentence = (
            f"Le premier indice de style est « {primary} », mais il ne se répète pas "
            "assez dans le morceau pour être considéré comme stable."
        )
        status = "low_confidence"
    else:
        genre_sentence = "Le style musical n'a pas pu être établi de façon suffisamment stable."
        status = "insufficient_evidence"

    singing_seconds = float(singing.get("total_duration_seconds", 0) or 0)
    if singing_seconds > 0:
        vocal_sentence = (
            f"FireRedVAD détecte des passages chantés sur environ {singing_seconds:.1f} secondes. "
            "Ce détecteur confirme la présence du chant, mais ne détermine ni le timbre, "
            "ni le registre, ni le genre vocal."
        )
    else:
        vocal_sentence = (
            "La présence de chant n'est pas confirmée par FireRedVAD ; cela ne prouve pas "
            "à lui seul que le morceau est instrumental."
        )

    if confirmed_instruments:
        names = [str(item.get("name", "")).strip().casefold() for item in confirmed_instruments]
        names = [name for name in names if name]
        instrument_sentence = (
            "Les candidats instrumentaux les plus récurrents sont : "
            + ", ".join(names[:4])
            + ". Ils restent des indices automatiques, pas une transcription des pistes."
        )
    else:
        instrument_sentence = (
            "Aucun instrument n'est retenu comme confirmé : le classifieur actuel a été entraîné "
            "sur des extraits centrés sur un instrument et sa fiabilité sur un mix complet "
            "n'est pas validée."
        )

    text = (
        f"Analyse ARTIST DNA — durée : {duration_label}. "
        f"{genre_sentence} {vocal_sentence} {instrument_sentence} "
        "Le sous-genre, la couleur sonore détaillée et la signature artistique restent à confirmer "
        "tant que des modèles adaptés et évalués sur des mixages complets ne sont pas disponibles."
    )
    return {"status": status, "text": text}
