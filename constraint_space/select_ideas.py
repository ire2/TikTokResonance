def select_ideas(ideas, policy):
    """
    Select ideas based on the experiment policy.

    Args:
        ideas: List of idea evaluation results.
        policy: Experiment policy dict.

    Returns:
        selected_ideas: List of selected idea IDs.
    """

    approved = []
    experiments = []
    rejected = []

    for e in ideas:
        decision = e["result"]["decision"]

        if decision == "pass":
            approved.append(e)
        elif decision == "warn":
            experiments.append(e)
        else:
            rejected.append(e)

    # Enforce experiment budget
    max_experiments = policy.get("max_off_pattern", 0)
    experiments = experiments[:max_experiments]

    return {
        "approved": approved,
        "experiments": experiments,
        "rejected": rejected,
    }
