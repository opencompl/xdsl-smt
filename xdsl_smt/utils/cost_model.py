import math

from xdsl_smt.utils.compare_result import CompareResult


def sound_and_precise_cost(res: CompareResult) -> float:
    a = 1
    b = 32
    sound = res.get_sound_prop()
    dis = res.get_unsolved_edit_dis_avg() / (res.bitwidth * 2)
    return (a * (1 - sound) + b * dis) / (a + b)


def precise_cost(res: CompareResult) -> float:
    a = 0
    b = 1
    sound = res.get_sound_prop()
    dis = res.get_unsolved_edit_dis_avg() / (res.bitwidth * 2)
    return (a * (1 - sound) + b * dis) / (a + b)


def abduction_cost(res: CompareResult) -> float:
    a = 1
    b = 4
    sound = res.get_sound_prop()
    dis = res.get_unsolved_edit_dis_avg() / (res.bitwidth * 2)
    return (a * (1 - sound) + b * dis) / (a + b)


def decide(p: float, beta: float, current_cost: float, proposed_cost: float) -> bool:
    return beta * (current_cost - proposed_cost) > math.log(p)
