from typing import Dict, Any

# TODO (v1):
# Replace rule-based evaluation with learned scoring model.
# Candidate approaches:
# - Train ranking model on (creator, idea, outcome) triples
# - Use creator embeddings + idea embeddings
# - Optimize for downstream engagement + creator satisfaction
# This function intentionally defines the decision interface
# before learning is introduced.


def evaluate_idea_against_constraints(
    idea: Dict[str, Any],
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Deterministic idea evaluation (v0).

    Returns:
        {
            decision: pass | warn | reject,
            score: float,
            confidence_band: low | medium | high,
            reasons: List[str],
            signals: Dict[str, Any]
        }
    """

    reasons = []
    signals = {}
    off_pattern_format = False
    voice_mismatch = False
    text_mismatch = False

    # -----------------------------
    # Duration check
    # -----------------------------
    preferred = constraints["duration_constraints"]["preferred_sec"]
    idea_duration = idea["format_hypothesis"]["duration_sec"]

    if preferred is not None:
        delta = abs(idea_duration - preferred)
        signals["duration_delta"] = delta

        if delta > 15:
            reasons.append(
                f"Idea duration ({idea_duration}s) deviates from "
                f"creator preference ({preferred}s)."
            )

    # -----------------------------
    # Modality checks
    # -----------------------------
    modality = constraints["modality_constraints"]

    if idea["format_hypothesis"]["has_voice"] and modality.get("voice") == "low":
        reasons.append(
            "Idea relies on voice, but creator shows low voice usage."
        )
        voice_mismatch = True

    if idea["format_hypothesis"]["has_text"] and modality.get("text") == "low":
        reasons.append(
            "Idea relies on text, but creator shows low text usage."
        )
        text_mismatch = True

    # -----------------------------
    # Format checks
    # -----------------------------
    allowed_formats = constraints["format_constraints"]["allowed"]
    idea_format = idea["format_hypothesis"]["format"]

    if allowed_formats:
        if idea_format not in allowed_formats:
            reasons.append("experimental format")
            off_pattern_format = True

    # -----------------------------
    # Experiment policy
    # -----------------------------
    policy = constraints["experiment_policy"]

    if off_pattern_format and not policy["allow_experiments"]:
        reasons.append("Experimental formats are disallowed.")
        decision = "reject"

    else:
        # Simple v0 decision rule
        if len(reasons) == 0:
            decision = "pass"
        elif len(reasons) <= policy.get("max_off_pattern", 1):
            decision = "warn"
        else:
            decision = "reject"

    # -----------------------------
    # Score + confidence band
    # -----------------------------
    score = 0.8
    if off_pattern_format:
        score -= 0.08
    if voice_mismatch:
        score -= 0.06
    if text_mismatch:
        score -= 0.06
    duration_delta = signals.get("duration_delta")
    if duration_delta is not None:
        if duration_delta > 10:
            score -= 0.04
        elif duration_delta > 5:
            score -= 0.02

    score = max(0.0, min(1.0, round(score, 2)))

    if score >= 0.8:
        confidence_band = "high"
    elif score >= 0.6:
        confidence_band = "medium"
    else:
        confidence_band = "low"

    return {
        "decision": decision,
        "score": score,
        "confidence_band": confidence_band,
        "reasons": reasons,
        "signals": signals,
    }
