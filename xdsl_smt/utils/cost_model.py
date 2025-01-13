def compute_cost(soundness: float, precision: float) -> float:
    return 4 * (1 - soundness) ** 2 + (1 - precision)
    # return 1 / (soundness + 1e-3)


def compute_accept_rate(current_cost: float, proposed_cost: float) -> float:
    # return math.exp(-16 * (proposed_cost - current_cost))
    return 1 if proposed_cost <= current_cost else 0
