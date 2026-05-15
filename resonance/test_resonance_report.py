from resonance.resonance_report import build_resonance_report


def _resonance_payload(format_alignment, dialogue_affinity=1.0):
    return {
        "semantic_alignment": 0.5,
        "dialogue_affinity": dialogue_affinity,
        "format_alignment": format_alignment,
        "evidence": [],
    }


def test_format_match_uses_format_alignment_not_dialogue_affinity():
    report = build_resonance_report(
        "test idea",
        _resonance_payload(format_alignment=0.3, dialogue_affinity=1.0),
    )

    assert report["interpretation"]["format_match"] == "weak"


def test_format_match_unknown_without_format_signal():
    report = build_resonance_report(
        "test idea",
        _resonance_payload(format_alignment=None, dialogue_affinity=1.0),
    )

    assert report["interpretation"]["format_match"] == "unknown"
