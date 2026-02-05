def build_constraints(reviewed_profile: dict):
    """
    Compile reviewed creator profile into a ConstraintSpace spec.
    """

    profile = reviewed_profile["profile"]

    dominant_formats = profile.get("observed_patterns", {}).get(
        "dominant_formats", []
    )

    avg_duration = profile.get("observed_patterns", {}).get(
        "avg_duration_sec", None
    )

    constraints = {
        "creator_id": reviewed_profile["creator_id"],
        "constraint_version": "v0.1",

        "format_constraints": {
            "allowed": dominant_formats,
        },

        "duration_constraints": {
            "preferred_sec": avg_duration,
        },

        "modality_constraints": profile.get("modality_bias", {}),

        "experiment_policy": {
            "allow_experiments": True,
            "max_off_pattern": 1,
        }
    }

    return constraints
