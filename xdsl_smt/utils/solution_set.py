from __future__ import annotations
from typing import Callable

from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.func import FuncOp, CallOp, ReturnOp

from xdsl_smt.utils.compare_result import CompareResult
from abc import ABC, abstractmethod
import logging


def rename_functions(lst: list[FuncOp], prefix: str) -> list[str]:
    func_names: list[str] = []
    for i, func in enumerate(lst):
        func_names.append(prefix + str(i))
        func.sym_name = StringAttr(func_names[-1])
    return func_names


"""
This class is an abstract class for maintaining solutions.
It supports to generate the meet of solutions
"""


class SolutionSet(ABC):
    solutions_size: int
    solutions: list[FuncOp]
    solution_names: list[str]
    solution_srcs: list[str]
    lower_to_cpp: Callable[[FuncOp], str]

    """
    list of name of transfer functions
    list of transfer functions
    list of name of reference functions
    list of reference functions
    """
    eval_func: Callable[
        [list[str], list[str], list[str], list[str]], list[CompareResult]
    ]

    def __init__(
        self,
        initial_solutions: list[FuncOp],
        lower_to_cpp: Callable[[FuncOp], str],
        eval_func: Callable[
            [list[str], list[str], list[str], list[str]], list[CompareResult]
        ],
    ):
        self.solutions = initial_solutions
        self.solutions_size = len(initial_solutions)
        self.lower_to_cpp = lower_to_cpp
        self.eval_func = eval_func
        self.solution_names = rename_functions(self.solutions, "partial_solution_")
        self.solution_srcs = [self.lower_to_cpp(func) for func in self.solutions]

    @abstractmethod
    def construct_new_solution_set(self, new_candidates: list[FuncOp]) -> SolutionSet:
        ...

    def has_solution(self) -> bool:
        return self.solutions_size != 0

    def generate_solution(self) -> FuncOp:
        assert self.has_solution()
        solutions = self.solutions
        result = FuncOp("solution", solutions[0].function_type)
        result_type = result.function_type.outputs.data
        part_result: list[CallOp] = []
        for ith, func in enumerate(solutions):
            cur_func_name = "partial_solution_" + str(ith)
            func.sym_name = StringAttr("partial_solution_" + str(ith))
            part_result.append(CallOp(cur_func_name, result.args, result_type))
        if len(part_result) == 1:
            result.body.block.add_ops(part_result + [ReturnOp(part_result[-1])])
        else:
            meet_result: list[CallOp] = [
                CallOp(
                    "meet",
                    [part_result[0], part_result[1]],
                    result_type,
                )
            ]
            for i in range(2, len(part_result)):
                meet_result.append(
                    CallOp(
                        "meet",
                        [meet_result[-1], part_result[i]],
                        result_type,
                    )
                )
            result.body.block.add_ops(
                part_result + meet_result + [ReturnOp(meet_result[-1])]
            )
        return result

    def generate_solution_and_cpp(self) -> tuple[FuncOp, str]:
        final_solution = self.generate_solution()
        solution_str = ""
        for src in self.solution_srcs:
            solution_str += src
            solution_str += "\n"
        solution_str += self.lower_to_cpp(final_solution)
        solution_str += "\n"
        return final_solution, solution_str


"""
This class maintains a list of solutions with a specified size
"""


class SizedSolutionSet(SolutionSet):
    size: int

    def __init__(
        self,
        size: int,
        initial_solutions: list[FuncOp],
        lower_to_cpp: Callable[[FuncOp], str],
        eval_func: Callable[
            [list[str], list[str], list[str], list[str]], list[CompareResult]
        ],
    ):
        super().__init__(initial_solutions, lower_to_cpp, eval_func)
        self.size = size

    def construct_new_solution_set(self, new_candidates: list[FuncOp]) -> SolutionSet:
        candidates = self.solutions + new_candidates

        if len(candidates) <= self.size:
            return SizedSolutionSet(
                self.size, candidates.copy(), self.lower_to_cpp, self.eval_func
            )

        candidates_names = rename_functions(candidates, "part_solution_")
        candidate_srcs: list[str] = [self.lower_to_cpp(ele) for ele in candidates]

        ref_funcs: list[FuncOp] = []
        ref_func_names: list[str] = []
        ref_func_srcs: list[str] = []

        # First select a function with maximal precise
        result = self.eval_func(
            candidates_names, candidate_srcs, ref_func_names, ref_func_srcs
        )
        index = 0
        num_exacts = 0
        cost = 2
        for i in range(len(result)):
            if result[i].exacts > num_exacts:
                index = i
                num_exacts = result[i].exacts
                cost = result[i].get_cost()
            elif result[i].exacts == num_exacts and result[i].get_cost() > cost:
                index = i
                cost = result[i].get_cost()

        ref_funcs.append(candidates.pop(index))
        ref_func_names.append(candidates_names.pop(index))
        ref_func_srcs.append(candidate_srcs.pop(index))

        # Greedy select all subsequent functions
        for _ in range(1, self.size + 1):
            index = 0
            num_exacts = 0
            cost = 2
            result = self.eval_func(
                candidates_names, candidate_srcs, ref_func_names, ref_func_srcs
            )
            for ith_result in range(len(result)):
                if result[ith_result].unsolved_exacts > num_exacts:
                    index = ith_result
                    num_exacts = result[ith_result].unsolved_exacts
                    cost = result[ith_result].get_cost()
                elif (
                    result[ith_result].unsolved_exacts == num_exacts
                    and cost > result[ith_result].get_cost()
                ):
                    index = ith_result
                    cost = result[ith_result].get_cost()
            ref_funcs.append(candidates.pop(index))
            ref_func_names.append(candidates_names.pop(index))
            ref_func_srcs.append(candidate_srcs.pop(index))

        return SizedSolutionSet(self.size, ref_funcs, self.lower_to_cpp, self.eval_func)


"""
This class maintains a list of solutions without a specified size
"""


class UnsizedSolutionSet(SolutionSet):
    logger: logging.Logger

    def __init__(
        self,
        initial_solutions: list[FuncOp],
        lower_to_cpp: Callable[[FuncOp], str],
        eval_func: Callable[
            [list[str], list[str], list[str], list[str]], list[CompareResult]
        ],
        logger: logging.Logger,
    ):
        super().__init__(initial_solutions, lower_to_cpp, eval_func)
        self.logger = logger

    def construct_new_solution_set(self, new_candidates: list[FuncOp]) -> SolutionSet:
        cur_most_e: float = 0
        candidate_names = rename_functions(new_candidates, "candidates_")
        self.logger.info(f"Size of candidates: {len(new_candidates)}")
        for i, func in enumerate(new_candidates):
            cpp_code = self.lower_to_cpp(func)

            cmp_results: list[CompareResult] = self.eval_func(
                [candidate_names[i]],
                [cpp_code],
                self.solution_names,
                self.solution_srcs,
            )
            if cmp_results[0].exacts > cur_most_e:
                self.logger.info(
                    f"Add a new transformer {i}. Exact: {cmp_results[0].get_exact_prop() * 100:.2f}%, Precision: {cmp_results[0].get_bitwise_precision() * 100:.2f}%"
                )
                self.logger.debug(cmp_results[0])
                cur_most_e = cmp_results[0].exacts
                self.solutions.append(func)
                self.solution_names.append(candidate_names[i])
                self.solution_srcs.append(cpp_code)
                self.solutions_size += 1

        self.logger.info(f"Size of the sound set: {self.solutions_size}")

        if cur_most_e == 0:
            self.logger.info(f"No improvement in the last one iteration!")

        # Remove redundant transformers
        i = 0
        while i < self.solutions_size:
            cmp_results: list[CompareResult] = self.eval_func(
                [self.solution_names[i]],
                [self.solution_srcs[i]],
                self.solution_names[:i] + self.solution_names[i + 1 :],
                self.solution_srcs[:i] + self.solution_srcs[i + 1 :],
            )
            if cmp_results[0].unsolved_exacts == 0:
                del self.solutions[i]
                del self.solution_names[i]
                del self.solution_srcs[i]
                self.solutions_size -= 1
            else:
                i += 1

        return UnsizedSolutionSet(
            self.solutions.copy(), self.lower_to_cpp, self.eval_func, self.logger
        )
