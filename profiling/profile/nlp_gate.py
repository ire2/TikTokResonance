def compute_nlp_gate(visual_summary: dict) -> dict:
    r = visual_summary.get("avg_face_presence_ratio")

    if r is None:
        return {"weight": 0.5, "confidence": "low", "reason": "insufficient CV data"}

    if r < 0.2:
        return {"weight": 0.0, "confidence": "high", "reason": "faces rarely present"}

    if r < 0.6:
        return {"weight": 0.5, "confidence": "medium", "reason": "mixed visual structure"}

    return {"weight": 1.0, "confidence": "high", "reason": "consistent talking-head presence"}
