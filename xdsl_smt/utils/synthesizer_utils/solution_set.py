from __future__ import annotations

import io
from typing import Callable

from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp, CallOp, ReturnOp
from xdsl.ir import Operation

from xdsl_smt.utils.synthesizer_utils.compare_result import CompareResult
from abc import ABC, abstractmethod
import logging
from xdsl_smt.utils.synthesizer_utils.verifier_utils import verify_transfer_function

from xdsl_smt.utils.synthesizer_utils.function_with_condition import (
    FunctionWithCondition,
)
from xdsl_smt.utils.synthesizer_utils.synthesizer_context import SynthesizerContext


def rename_functions(lst: list[FunctionWithCondition], prefix: str) -> list[str]:
    func_names: list[str] = []
    for i, func in enumerate(lst):
        func_names.append(prefix + str(i))
        func.set_func_name(func_names[-1])
    return func_names


def verify_function(
    func: FunctionWithCondition,
    concrete_op: FuncOp,
    helper_funcs: list[FuncOp],
    ctx: MLContext,
) -> int:
    cur_helper = [func.func]
    if func.cond is not None:
        cur_helper.append(func.cond)
    return verify_transfer_function(
        func.get_function(), cur_helper + helper_funcs, ctx, 32
    )


"""
This class is an abstract class for maintaining solutions.
It supports to generate the meet of solutions
"""


class SolutionSet(ABC):
    solutions_size: int
    solutions: list[FunctionWithCondition]
    precise_set: list[FuncOp]
    lower_to_cpp: Callable[[FuncOp], str]
    eliminate_dead_code: Callable[[FuncOp], FuncOp]
    is_perfect: bool

    """
    list of name of transfer functions
    list of transfer functions
    list of name of base functions
    list of base functions
    """
    eval_func: Callable[
        [list[FunctionWithCondition], list[FunctionWithCondition]], list[CompareResult]
    ]
    logger: logging.Logger

    def __init__(
        self,
        initial_solutions: list[FunctionWithCondition],
        lower_to_cpp: Callable[[FuncOp], str],
        eliminate_dead_code: Callable[[FuncOp], FuncOp],
        eval_func: Callable[
            [
                list[FunctionWithCondition],
                list[FunctionWithCondition],
            ],
            list[CompareResult],
        ],
        logger: logging.Logger,
        is_perfect: bool = False,
    ):
        rename_functions(initial_solutions, "partial_solution_")
        self.solutions = initial_solutions
        self.solutions_size = len(initial_solutions)
        self.lower_to_cpp = lower_to_cpp
        self.eliminate_dead_code = eliminate_dead_code
        self.eval_func = eval_func
        self.logger = logger
        self.precise_set = []
        self.is_perfect = is_perfect

    def eval_improve(
        self, transfers: list[FunctionWithCondition]
    ) -> list[CompareResult]:
        return self.eval_func(transfers, self.solutions)

    @abstractmethod
    def construct_new_solution_set(
        self,
        new_candidates_sp: list[FunctionWithCondition],
        new_candidates_p: list[FuncOp],
        new_candidates_c: list[FunctionWithCondition],
        # Parameters used by SMT verifier
        concrete_op: FuncOp,
        helper_funcs: list[FuncOp],
        ctx: MLContext,
    ) -> SolutionSet:
        ...

    def has_solution(self) -> bool:
        return self.solutions_size != 0

    def generate_solution(self) -> tuple[FuncOp, list[FuncOp]]:
        assert self.has_solution()
        solutions = self.solutions
        result = FuncOp("solution", solutions[0].func.function_type)
        result_type = result.function_type.outputs.data
        part_result: list[CallOp] = []
        part_solution_funcs: list[FuncOp] = []
        for ith, func_with_cond in enumerate(solutions):
            cur_func_name = "partial_solution_" + str(ith)
            func_with_cond.set_func_name(cur_func_name)
            part_solution_funcs.append(func_with_cond.get_function())
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
        return result, part_solution_funcs

    def generate_solution_and_cpp(self) -> tuple[ModuleOp, str]:
        final_solution, part_solutions = self.generate_solution()
        function_lst: list[FuncOp] = []
        solution_str = ""
        for sol in self.solutions:
            func_body = self.eliminate_dead_code(sol.func)
            function_lst.append(func_body)
            solution_str += self.lower_to_cpp(func_body)
            solution_str += "\n"
            if sol.cond is not None:
                func_cond = self.eliminate_dead_code(sol.cond)
                function_lst.append(func_cond)
                solution_str += self.lower_to_cpp(func_cond)
                solution_str += "\n"

        for sol in part_solutions:
            solution_str += self.lower_to_cpp(sol)
            solution_str += "\n"
        solution_str += self.lower_to_cpp(final_solution)
        solution_str += "\n"

        function_lst += part_solutions
        function_lst.append(final_solution)
        final_module = ModuleOp([])
        final_module.body.block.add_ops(function_lst)
        return final_module, solution_str

    def remove_unsound_solutions(
        self, concrete_op: FuncOp, helper_funcs: list[FuncOp], ctx: MLContext
    ):
        unsound_lst: list[int] = []
        for i, sol in enumerate(self.solutions):
            cur_helper = [sol.func]
            if sol.cond is not None:
                cur_helper.append(sol.cond)
            if not verify_transfer_function(
                sol.get_function(), cur_helper + helper_funcs, ctx, 16
            ):
                unsound_lst.append(i)
        for unsound_idx in unsound_lst[::-1]:
            self.solutions.pop(unsound_idx)
            self.solutions_size -= 1
            self.is_perfect = False


"""
This class maintains a list of solutions with a specified size
"""


class SizedSolutionSet(SolutionSet):
    size: int

    def __init__(
        self,
        size: int,
        initial_solutions: list[FunctionWithCondition],
        lower_to_cpp: Callable[[FuncOp], str],
        eliminate_dead_code: Callable[[FuncOp], FuncOp],
        eval_func_with_cond: Callable[
            [
                list[FunctionWithCondition],
                list[FunctionWithCondition],
            ],
            list[CompareResult],
        ],
        logger: logging.Logger,
        is_perfect: bool = False,
    ):
        super().__init__(
            initial_solutions,
            lower_to_cpp,
            eliminate_dead_code,
            eval_func_with_cond,
            logger,
            is_perfect,
        )
        self.size = size

    def construct_new_solution_set(
        self,
        new_candidates_sp: list[FunctionWithCondition],
        new_candidates_p: list[FuncOp],
        new_candidates_c: list[FunctionWithCondition],
        concrete_op: FuncOp,
        helper_funcs: list[FuncOp],
        ctx: MLContext,
    ) -> SolutionSet:
        candidates = self.solutions + new_candidates_sp
        if len(candidates) <= self.size:
            return SizedSolutionSet(
                self.size,
                candidates.copy(),
                self.lower_to_cpp,
                self.eliminate_dead_code,
                self.eval_func,
                self.logger,
            )
        rename_functions(candidates, "part_solution_")
        ref_funcs: list[FunctionWithCondition] = []

        # First select a function with maximal precise
        result: list[CompareResult] = self.eval_func(candidates, ref_funcs)
        index = 0
        num_exacts = 0
        # cost = 2
        for i in range(len(result)):
            if result[i].exacts > num_exacts:
                index = i
                num_exacts = result[i].exacts
            # temporarily comment this out since (1) now the cost depends on both synthcontext and cmpresult (2) I think #exacts is enough to rank tfs
            #     cost = result[i].get_cost()
            # elif result[i].exacts == num_exacts and result[i].get_cost() > cost:
            #     index = i
            #     cost = result[i].get_cost()

        ref_funcs.append(candidates.pop(index))

        is_perfect = False
        # Greedy select all subsequent functions
        for _ in range(1, self.size + 1):
            index = 0
            num_exacts = 0
            # cost = 2
            result: list[CompareResult] = self.eval_func(candidates, ref_funcs)
            for ith_result in range(len(result)):
                if result[ith_result].unsolved_cases == 0:
                    is_perfect = True
                    break
                if result[ith_result].unsolved_exacts > num_exacts:
                    index = ith_result
                    num_exacts = result[ith_result].unsolved_exacts
            # Xuanyu: temporarily comment this out since (1) now the cost depends on both mcmc_sampler and cmp_result (2) I think #exacts is enough to rank tfs
            #     cost = result[ith_result].get_cost()
            # elif (
            #     result[ith_result].unsolved_exacts == num_exacts
            #     and cost > result[ith_result].get_cost()
            # ):
            #     index = ith_result
            #     cost = result[ith_result].get_cost()
            ref_funcs.append(candidates.pop(index))

        return SizedSolutionSet(
            self.size,
            ref_funcs,
            self.lower_to_cpp,
            self.eliminate_dead_code,
            self.eval_func,
            self.logger,
            is_perfect,
        )


"""
This class maintains a list of solutions without a specified size
"""


class UnsizedSolutionSet(SolutionSet):
    def __init__(
        self,
        initial_solutions: list[FunctionWithCondition],
        lower_to_cpp: Callable[[FuncOp], str],
        eval_func_with_cond: Callable[
            [
                list[FunctionWithCondition],
                list[FunctionWithCondition],
            ],
            list[CompareResult],
        ],
        logger: logging.Logger,
        eliminate_dead_code: Callable[[FuncOp], FuncOp],
        is_perfect: bool = False,
    ):
        super().__init__(
            initial_solutions,
            lower_to_cpp,
            eliminate_dead_code,
            eval_func_with_cond,
            logger,
            is_perfect,
        )

    def construct_new_solution_set(
        self,
        new_candidates_sp: list[FunctionWithCondition],
        new_candidates_p: list[FuncOp],
        new_candidates_c: list[FunctionWithCondition],
        concrete_op: FuncOp,
        helper_funcs: list[FuncOp],
        ctx: MLContext,
    ) -> SolutionSet:
        candidates = self.solutions + new_candidates_sp + new_candidates_c
        rename_functions(candidates, "part_solution_")
        self.logger.info(f"Size of new candidates: {len(new_candidates_sp)}")
        self.logger.info(f"Size of new conditional candidates: {len(new_candidates_c)}")
        self.logger.info(f"Size of solutions: {len(candidates)}")
        self.solutions = []
        self.logger.info("Reset solution set...")
        num_cond_solutions = 0
        while len(candidates) > 0:
            index = 0
            most_unsol_e = 0
            result = self.eval_improve(candidates)
            for ith_result in range(len(result)):
                if result[ith_result].unsolved_cases == 0:
                    self.is_perfect = True
                    break
                if result[ith_result].unsolved_exacts > most_unsol_e:
                    index = ith_result
                    most_unsol_e = result[ith_result].unsolved_exacts
            if most_unsol_e == 0:
                break

            unsound_bit = verify_function(
                candidates[index], concrete_op, helper_funcs, ctx
            )
            if unsound_bit != 0:
                self.logger.info(f"Skip a unsound function at bit width {unsound_bit}")
                if unsound_bit == 4:
                    func_str, helper_str = candidates[index].get_function_str(
                        self.lower_to_cpp
                    )
                    func_op = candidates[index].get_function()
                    for s in helper_str:
                        self.logger.critical(s + "\n")
                    self.logger.critical(func_str)
                    str_output = io.StringIO()
                    print(candidates[index].func, file=str_output)
                    if candidates[index].cond is not None:
                        print(candidates[index].cond, file=str_output)
                    print(func_op, file=str_output)
                    func_op_str = str_output.getvalue()
                    self.logger.error(func_op_str)
                    exit(0)
                candidates.pop(index)
                continue

            if candidates[index] in new_candidates_sp:
                log_str = "Add a new transformer"
            elif candidates[index] in new_candidates_c:
                log_str = "Add a new transformer (cond)"
                num_cond_solutions += 1
            else:
                if candidates[index].cond is None:
                    log_str = "Add a existing transformer"
                else:
                    log_str = "Add a existing transformer (cond)"
                    num_cond_solutions += 1
            self.logger.info(
                f"{log_str}. After adding, Exact: {result[index].get_exact_prop() * 100:.2f}%, Precision: {result[index].get_bitwise_precision() * 100:.2f}%"
            )
            self.solutions.append(candidates.pop(index))

        self.logger.info(
            f"The number of solutions after reseting: {len(self.solutions)}"
        )
        self.logger.info(f"The number of conditional solutions: {num_cond_solutions}")

        if self.is_perfect:
            return self

        precise_candidates = self.precise_set + new_candidates_p
        precise_candidates_to_eval = [
            FunctionWithCondition(f.clone()) for f in precise_candidates
        ]
        rename_functions(precise_candidates_to_eval, "precise_candidates_")
        result = self.eval_improve(precise_candidates_to_eval)

        sorted_pairs = sorted(
            zip(precise_candidates, result),
            reverse=True,
            key=lambda pair: pair[1].unsolved_exacts,
        )
        K = 15
        top_k = sorted_pairs[:K]
        self.logger.info(f"Top {K} Precise candidates:")
        self.precise_set = []
        for cand, res in top_k:
            self.logger.info(
                f"\tunsolved_exact: {res.get_unsolved_exact_prop() * 100:.2f}%, sound: {res.get_sound_prop() * 100:.2f}%"
            )
            self.precise_set.append(cand)
        self.solutions_size = len(self.solutions)
        return self

    """
    Set weights in context according to the frequencies of each DSL operation that appear in func in solution set
    """

    def learn_weights(self, context: SynthesizerContext):
        freq_i1: dict[type[Operation], int] = {}
        freq_int: dict[type[Operation], int] = {}

        def add_another_dict(
            dict1: dict[type[Operation], int], dict2: dict[type[Operation], int]
        ):
            for k, v in dict2.items():
                dict1[k] = dict1.get(k, 0) + v

        self.logger.info(f"Improvement by each individual function")
        for i in range(len(self.solutions)):
            cmp_results: list[CompareResult] = self.eval_func(
                [self.solutions[i]],
                self.solutions[:i] + self.solutions[i + 1 :],
            )
            res = cmp_results[0]
            self.logger.info(
                f"\tfunc {i}: #exact {res.exacts - res.unsolved_exacts} -> {res.exacts}, new exact%: {res.get_new_exact_prop()}, prec: {res.base_edit_dis} -> {res.edit_dis}, prec improve%: {res.get_prec_improve_avg()}, cond?: {self.solutions[i].cond is not None}"
            )
            if res.get_new_exact_prop() > 0.005:
                d_int, d_i1 = SynthesizerContext.count_op_frequency(
                    self.eliminate_dead_code(self.solutions[i].func)
                )
                add_another_dict(freq_int, d_int)
                add_another_dict(freq_i1, d_i1)

        context.update_i1_weights(freq_i1)
        context.update_int_weights(freq_int)

        self.logger.info("Current Weights:")
        for key, value in context.int_weights.items():
            self.logger.info(f"\t{key}: {value}")
