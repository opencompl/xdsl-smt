import math


def compute_cost(soundness: float, precision: float) -> float:
    # if soundness == 1:
    #     return 1 - precision
    # else:
    #     return 2 - soundness
    return 16 * (1 - soundness) ** 2 + 1 - precision


def decide(p: float, beta: float, current_cost: float, proposed_cost: float) -> bool:
    # return math.exp(-16 * (proposed_cost - current_cost))
    # return 1 if proposed_cost <= current_cost else 0

    return beta * (current_cost - proposed_cost) > math.log(p)
