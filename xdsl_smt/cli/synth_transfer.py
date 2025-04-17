import argparse
import logging
import os.path
import time
from typing import cast, Callable

from xdsl.context import MLContext
from xdsl.parser import Parser

from io import StringIO

from xdsl.utils.hints import isa

from xdsl_smt.utils.synthesizer_utils.compare_result import CompareResult
from ..dialects.smt_dialect import (
    SMTDialect,
)
from ..dialects.smt_bitvector_dialect import (
    SMTBitVectorDialect,
    ConstantOp,
)
from xdsl_smt.dialects.transfer import (
    TransIntegerType,
)
from ..dialects.index_dialect import Index
from ..dialects.smt_utils_dialect import SMTUtilsDialect
import xdsl_smt.eval_engine.eval as eval_engine
from xdsl.dialects.builtin import (
    Builtin,
    ModuleOp,
    IntegerAttr,
    IntegerType,
    i1,
    FunctionType,
    ArrayAttr,
    StringAttr,
)
from xdsl.dialects.func import Func, FuncOp, ReturnOp, CallOp
from ..dialects.transfer import Transfer
from xdsl.dialects.arith import Arith, ConstantOp
from xdsl.dialects.comb import Comb
from xdsl.dialects.hw import HW
from ..passes.transfer_dead_code_elimination import TransferDeadCodeElimination

import xdsl.dialects.comb as comb
from xdsl.ir import Operation

from ..passes.transfer_lower import LowerToCpp

from xdsl_smt.semantics.comb_semantics import comb_semantics
import sys as sys

from xdsl_smt.utils.synthesizer_utils.cost_model import (
    decide,
    sound_and_precise_cost,
    precise_cost,
    abduction_cost,
)
from xdsl_smt.utils.synthesizer_utils.function_with_condition import (
    FunctionWithCondition,
)
from xdsl_smt.utils.synthesizer_utils.log_utils import (
    setup_loggers,
    print_set_of_funcs_to_file,
)
from xdsl_smt.utils.synthesizer_utils.mcmc_sampler import MCMCSampler
from xdsl_smt.utils.synthesizer_utils.mutation_program import MutationProgram
from xdsl_smt.utils.synthesizer_utils.solution_set import (
    SolutionSet,
    UnsizedSolutionSet,
    SizedSolutionSet,
)
from xdsl_smt.utils.synthesizer_utils.synthesizer_context import SynthesizerContext
from xdsl_smt.utils.synthesizer_utils.random import Random

# from ..utils.visualize import print_figure


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "transfer_functions", type=str, nargs="?", help="path to the transfer functions"
    )
    arg_parser.add_argument(
        "-random_file", type=str, nargs="?", help="the file includes all random numbers"
    )
    arg_parser.add_argument(
        "-random_seed", type=int, nargs="?", help="specify the random seed"
    )
    arg_parser.add_argument(
        "-program_length",
        type=int,
        nargs="?",
        help="Specify the maximal length of synthesized program. 40 by default.",
    )
    arg_parser.add_argument(
        "-total_rounds",
        type=int,
        nargs="?",
        help="Specify the number of rounds the synthesizer should run. 1000 by default.",
    )
    arg_parser.add_argument(
        "-num_programs",
        type=int,
        nargs="?",
        help="Specify the number of programs that runs at every round. 100 by default.",
    )
    arg_parser.add_argument(
        "-inv_temp",
        type=int,
        nargs="?",
        help="Inverse temperature \\beta for MCMC. The larger the value is, the lower the probability of accepting a program with a higher cost. 200 by default. "
        "E.g., MCMC has a 1/2 probability of accepting a program with a cost 1/beta higher. ",
    )
    arg_parser.add_argument(
        "-bitwidth",
        type=int,
        nargs="?",
        help="Specify the bitwidth of the evaluation engine",
    )
    arg_parser.add_argument(
        "-solution_size",
        type=int,
        nargs="?",
        help="Specify the size of solution set",
    )
    arg_parser.add_argument(
        "-num_iters",
        type=int,
        nargs="?",
        help="Specify the number of iterations of the synthesizer needs to run",
    )
    arg_parser.add_argument(
        "-weighted_dsl",
        action="store_true",
        help="Learn weights for each DSL operations from previous for future iterations.",
    )
    arg_parser.add_argument(
        "-condition_length",
        type=int,
        nargs="?",
        help="Specify the maximal length of synthesized abduction. 6 by default.",
    )
    arg_parser.add_argument(
        "-num_abd_procs",
        type=int,
        nargs="?",
        help="Specify the number of mcmc processes that used for abduction. It should be less than num_programs. 0 by default (which means abduction is disabled).",
    )
    arg_parser.add_argument(
        "-outputs_folder",
        type=str,
        nargs="?",
        help="Output folder for saving logs",
    )


def parse_file(ctx: MLContext, file: str | None) -> Operation:
    if file is None:
        f = sys.stdin
        file = "<stdin>"
    else:
        f = open(file)

    parser = Parser(ctx, f.read(), file)
    module = parser.parse_op()
    return module


def is_transfer_function(func: FuncOp) -> bool:
    return "applied_to" in func.attributes


def is_forward(func: FuncOp) -> bool:
    if "is_forward" in func.attributes:
        forward = func.attributes["is_forward"]
        assert isinstance(forward, IntegerAttr)
        return forward.value.data == 1
    return False


def get_concrete_function(
    concrete_op_name: str, width: int, extra: int | None
) -> FuncOp:
    # iterate all semantics and find corresponding comb operation
    result = None
    for k in comb_semantics.keys():
        if k.name == concrete_op_name:
            # generate a function with the only comb operation
            # for now, we only handle binary operations and mux
            intTy = IntegerType(width)
            transIntTy = TransIntegerType()
            func_name = "concrete_op"

            if concrete_op_name == "comb.mux":
                funcTy = FunctionType.from_lists([i1, intTy, intTy], [intTy])
                result = FuncOp(func_name, funcTy)
                combOp = k(*result.args)
            elif concrete_op_name == "comb.icmp":
                funcTy = FunctionType.from_lists([intTy, intTy], [i1])
                result = FuncOp(func_name, funcTy)
                assert extra is not None
                combOp = comb.ICmpOp(result.args[0], result.args[1], extra)
            elif concrete_op_name == "comb.concat":
                funcTy = FunctionType.from_lists(
                    [intTy, intTy], [IntegerType(width * 2)]
                )
                result = FuncOp(func_name, funcTy)
                combOp = comb.ConcatOp.from_int_values(result.args)
            else:
                funcTy = FunctionType.from_lists([transIntTy, transIntTy], [transIntTy])
                result = FuncOp(func_name, funcTy)
                if issubclass(k, comb.VariadicCombOperation):
                    combOp = k.create(operands=result.args, result_types=[intTy])
                else:
                    combOp = k(*result.args)

            assert isinstance(combOp, Operation)
            returnOp = ReturnOp(combOp.results[0])
            result.body.block.add_ops([combOp, returnOp])
    assert result is not None and (
        "Cannot find the concrete function for" + concrete_op_name
    )
    return result


def print_concrete_function_to_cpp(func: FuncOp) -> str:
    sio = StringIO()
    LowerToCpp(sio, True).apply(ctx, cast(ModuleOp, func))
    return sio.getvalue()


def print_to_cpp(func: FuncOp) -> str:
    sio = StringIO()
    LowerToCpp(sio).apply(ctx, cast(ModuleOp, func))
    return sio.getvalue()


def get_default_op_constraint(concrete_func: FuncOp):
    cond_type = FunctionType.from_lists(concrete_func.function_type.inputs.data, [i1])
    func = FuncOp("op_constraint", cond_type)
    true_op: ConstantOp = ConstantOp(IntegerAttr.from_int_and_width(1, 1), i1)
    return_op = ReturnOp(true_op)
    func.body.block.add_ops([true_op, return_op])
    return func


SYNTH_WIDTH = 4
TEST_SET_SIZE = 1000
CONCRETE_VAL_PER_TEST_CASE = 10
PROGRAM_LENGTH = 40
CONDITION_LENGTH = 6
NUM_PROGRAMS = 100
INIT_COST = 1
TOTAL_ROUNDS = 10000
INV_TEMP = 200
SOLUTION_SIZE = 8
NUM_ITERS = 100
INSTANCE_CONSTRAINT = "getInstanceConstraint"
DOMAIN_CONSTRAINT = "getConstraint"
OP_CONSTRAINT = "op_constraint"
MEET_FUNC = "meet"
GET_TOP_FUNC = "getTop"
CONCRETE_OP_FUNC = "concrete_op"
get_top_func_op: FuncOp | None = None
ret_top_func: FunctionWithCondition
TMP_MODULE: list[ModuleOp] = []
ctx: MLContext

OUTPUTS_FOLDER = "outputs"
LOG_FILE = "synth.log"
VERBOSE = 1  # todo: make it a cmd line arg


def eliminate_dead_code(func: FuncOp) -> FuncOp:
    new_module = ModuleOp([func.clone()])
    TransferDeadCodeElimination().apply(ctx, new_module)
    func_op = new_module.ops.first
    assert isinstance(func_op, FuncOp)
    func_op.detach()
    return func_op


def is_base_function(func: FuncOp) -> bool:
    # if "is_base" in func.attributes:
    #     base = func.attributes["is_base"]
    #     assert isinstance(base, IntegerAttr)
    #     return base.value.data == -1
    # return False
    return func.sym_name.data.startswith("part_solution_")


def construct_top_func(transfer: FuncOp) -> FuncOp:
    func = FuncOp("top_transfer_function", transfer.function_type)
    func.attributes["applied_to"] = transfer.attributes["applied_to"]
    func.attributes["CPPCLASS"] = transfer.attributes["CPPCLASS"]
    func.attributes["is_forward"] = transfer.attributes["is_forward"]
    block = func.body.block
    args = func.args

    call_top_op = CallOp("getTop", [args[0]], func.function_type.outputs.data)
    assert len(call_top_op.results) == 1
    top_res = call_top_op.results[0]
    return_op = ReturnOp(top_res)
    block.add_ops([call_top_op, return_op])
    return func


def eval_transfer_func_helper(
    transfer: list[FunctionWithCondition],
    base: list[FunctionWithCondition],
    domain: eval_engine.AbstractDomain,
    bitwidth: int,
    helper_funcs: list[str],
) -> list[CompareResult]:
    """
    This function is a helper of eval_transfer_func that prints the mlir func as cpp code
    When transfer is [], this function fill it into [top]
    """

    if not transfer:
        transfer = [ret_top_func]
    transfer_func_names: list[str] = []
    transfer_func_srcs: list[str] = []
    helper_func_srcs: list[str] = []
    assert get_top_func_op is not None
    for fc in transfer:
        caller_str, helper_strs = fc.get_function_str(print_to_cpp, eliminate_dead_code)
        transfer_func_names.append(fc.func_name)
        transfer_func_srcs.append(caller_str)
        helper_func_srcs += helper_strs

    base_func_names: list[str] = []
    base_func_srcs: list[str] = []
    for fc in base:
        caller_str, helper_strs = fc.get_function_str(print_to_cpp, eliminate_dead_code)
        base_func_names.append(fc.func_name)
        base_func_srcs.append(caller_str)
        helper_func_srcs += helper_strs

    return eval_engine.eval_transfer_func(
        transfer_func_names,
        transfer_func_srcs,
        base_func_names,
        base_func_srcs,
        helper_funcs + helper_func_srcs,
        domain,
        bitwidth,
    )


"""
This function returns a simplified eval_func receiving transfer functions and base functions
"""


def solution_set_eval_func(
    domain: eval_engine.AbstractDomain,
    bitwidth: int,
    helper_funcs: list[str],
) -> Callable[
    [
        list[FunctionWithCondition],
        list[FunctionWithCondition],
    ],
    list[CompareResult],
]:
    return lambda transfer=list[FunctionWithCondition], base=list[
        FunctionWithCondition
    ]: (eval_transfer_func_helper(transfer, base, domain, bitwidth, helper_funcs))


"""
This function returns a simplified eval_func only receiving transfer names and sources
"""


def main_eval_func(
    base_transfers: list[FunctionWithCondition],
    domain: eval_engine.AbstractDomain,
    bitwidth: int,
    helper_funcs: list[str],
) -> Callable[[list[FunctionWithCondition]], list[CompareResult]]:
    return lambda transfers=list[FunctionWithCondition]: (
        eval_transfer_func_helper(
            transfers, base_transfers, domain, bitwidth, helper_funcs
        )
    )


def build_eval_list(
    mcmc_proposals: list[FuncOp],
    sp: range,
    p: range,
    c: range,
    prec_func_after_distribute: list[FuncOp],
) -> list[FunctionWithCondition]:
    """
    build the parameters of eval_transfer_func
    input:
    mcmc_proposals =  [ ..mcmc_sp.. , ..mcmc_p.. , ..mcmc_c.. ]
    output:
    funcs          =  [ ..mcmc_sp.. , ..mcmc_p.. ,..prec_set..]
    conds          =  [  nothing    ,  nothing   , ..mcmc_c.. ]
    """
    lst: list[FunctionWithCondition] = []
    for i in sp:
        fwc = FunctionWithCondition(mcmc_proposals[i])
        fwc.set_func_name(f"{mcmc_proposals[i].sym_name.data}{i}")
        lst.append(fwc)
    for i in p:
        fwc = FunctionWithCondition(mcmc_proposals[i])
        fwc.set_func_name(f"{mcmc_proposals[i].sym_name.data}{i}")
        lst.append(fwc)
    for i in c:
        prec_func = prec_func_after_distribute[i - c.start].clone()
        fwc = FunctionWithCondition(prec_func, mcmc_proposals[i])
        fwc.set_func_name(f"{prec_func.sym_name.data}_abd_{i}")
        lst.append(fwc)

    return lst


def mcmc_setup(
    solution_set: SolutionSet, num_abd_proc: int, num_programs: int
) -> tuple[range, range, range, int, list[FuncOp]]:
    """
    A mcmc sampler use one of 3 modes: sound & precise, precise, condition
    This function specify which mode should be used for each mcmc sampler
    For example, mcmc samplers with index in sp_range should use "sound&precise"
    """

    # p_size = num_abd_proc // 2
    # c_size = num_abd_proc // 2
    p_size = 0
    c_size = num_abd_proc
    sp_size = num_programs - p_size - c_size

    if len(solution_set.precise_set) == 0:
        sp_size += c_size
        c_size = 0

    sp_range = range(0, sp_size)
    p_range = range(sp_size, sp_size + p_size)
    c_range = range(sp_size + p_size, sp_size + p_size + c_size)

    prec_set_after_distribute: list[FuncOp] = []

    if c_size > 0:
        # Distribute the precise funcs into c_range
        prec_set_size = len(solution_set.precise_set)
        base_count = c_size // prec_set_size
        remainder = c_size % prec_set_size
        for i, item in enumerate(solution_set.precise_set):
            for _ in range(base_count + (1 if i < remainder else 0)):
                prec_set_after_distribute.append(item.clone())

    num_programs = sp_size + p_size + c_size

    return sp_range, p_range, c_range, num_programs, prec_set_after_distribute


"""
Given ith_iter, performs total_rounds mcmc sampling
"""


def synthesize_transfer_function(
    # Necessary items
    ith_iter: int,
    func: FuncOp,
    context_regular: SynthesizerContext,
    context_weighted: SynthesizerContext,
    context_cond: SynthesizerContext,
    random: Random,
    solution_set: SolutionSet,
    logger: logging.Logger,
    # Evalate transfer functions
    eval_func: Callable[[list[FunctionWithCondition]], list[CompareResult]],
    concrete_func: FuncOp,
    helper_funcs: list[FuncOp],
    ctx: MLContext,
    # Global arguments
    num_programs: int,
    program_length: int,
    cond_length: int,
    num_abd_procs: int,
    total_rounds: int,
    solution_size: int,
    inv_temp: int,
    outputs_folder: str,
) -> SolutionSet:
    mcmc_samplers: list[MCMCSampler] = []

    sp_range, p_range, c_range, num_programs, prec_set_after_distribute = mcmc_setup(
        solution_set, num_abd_procs, num_programs
    )
    sp_size = sp_range.stop - sp_range.start
    p_size = p_range.stop - p_range.start

    for i in range(num_programs):
        if i in sp_range:
            sampler = MCMCSampler(
                func,
                context_regular
                if i < (sp_range.start + sp_range.stop) // 2
                else context_weighted,
                sound_and_precise_cost,
                program_length,
                random_init_program=True,
            )
        elif i in p_range:
            sampler = MCMCSampler(
                func,
                context_regular
                if i < (p_range.start + p_range.stop) // 2
                else context_weighted,
                precise_cost,
                program_length,
                random_init_program=True,
            )
        else:
            sampler = MCMCSampler(
                func,
                context_cond,
                abduction_cost,
                cond_length,
                random_init_program=True,
                is_cond=True,
            )

        mcmc_samplers.append(sampler)

    # Get the cost of initial programs
    transfers: list[FuncOp] = []
    for i in range(num_programs):
        transfers.append(mcmc_samplers[i].get_current().clone())
    func_with_cond_lst = build_eval_list(
        transfers, sp_range, p_range, c_range, prec_set_after_distribute
    )

    if solution_size == 0:
        cmp_results: list[CompareResult] = solution_set.eval_improve(func_with_cond_lst)
    else:
        cmp_results: list[CompareResult] = eval_func(func_with_cond_lst)
    for i in range(num_programs):
        mcmc_samplers[i].current_cmp = cmp_results[i]

    cost_data: list[list[float]] = [
        [mcmc_samplers[i].compute_current_cost()] for i in range(num_programs)
    ]
    """
    These 3 lists store "good" transformers during the search
    """
    sound_most_exact_tfs: list[tuple[MutationProgram, CompareResult, int]] = []
    most_exact_tfs: list[tuple[MutationProgram, CompareResult, int]] = []
    lowest_cost_tfs: list[tuple[MutationProgram, CompareResult, int]] = []
    for i in range(num_programs):
        mcmc_samplers[i].current_cmp = cmp_results[i]
        sound_most_exact_tfs.append((mcmc_samplers[i].current, cmp_results[i], 0))
        most_exact_tfs.append((mcmc_samplers[i].current, cmp_results[i], 0))
        lowest_cost_tfs.append((mcmc_samplers[i].current, cmp_results[i], 0))
    # MCMC start
    logger.info(
        f"Iter {ith_iter}: Start {num_programs} MCMC. Each one is run for {total_rounds} steps..."
    )

    for rnd in range(total_rounds):
        transfers: list[FuncOp] = []
        for i in range(num_programs):
            _: float = mcmc_samplers[i].sample_next()
            proposed_solution = mcmc_samplers[i].get_proposed()
            assert proposed_solution is not None
            transfers.append(proposed_solution.clone())

        func_with_cond_lst = build_eval_list(
            transfers, sp_range, p_range, c_range, prec_set_after_distribute
        )

        start = time.time()
        if solution_size == 0:
            cmp_results: list[CompareResult] = solution_set.eval_improve(
                func_with_cond_lst
            )
        else:
            cmp_results: list[CompareResult] = eval_func(func_with_cond_lst)
        end = time.time()
        used_time = end - start

        for i in range(num_programs):
            spl_i = mcmc_samplers[i]
            res_i = cmp_results[i]
            proposed_cost = spl_i.compute_cost(res_i)
            current_cost = spl_i.compute_current_cost()
            p = random.random()
            decision = decide(p, inv_temp, current_cost, proposed_cost)
            if decision:
                spl_i.accept_proposed(res_i)
                assert spl_i.get_proposed() is None
                tmp_tuple = (spl_i.current, res_i, rnd)
                need_print = False
                # Update sound_most_exact_tfs
                if (
                    res_i.is_sound()
                    and res_i.exacts > sound_most_exact_tfs[i][1].exacts
                ):
                    sound_most_exact_tfs[i] = tmp_tuple
                    need_print = True
                # Update most_exact_tfs
                if res_i.unsolved_exacts > most_exact_tfs[i][1].unsolved_exacts:
                    most_exact_tfs[i] = tmp_tuple
                    need_print = True
                # Update lowest_cost_tfs
                if proposed_cost < spl_i.compute_cost(lowest_cost_tfs[i][1]):
                    lowest_cost_tfs[i] = tmp_tuple
                    need_print = True

                # disable it temporarily
                if need_print:
                    pass
                    # print_func_to_file(
                    #     mcmc_samplers[i].current_cmp, eliminate_dead_code(mcmc_samplers[i].current.func), iter, rnd, i, OUTPUTS_FOLDER
                    # )
            else:
                spl_i.reject_proposed()
                pass
        for i in range(num_programs):
            res = mcmc_samplers[i].current_cmp
            res_cost = mcmc_samplers[i].compute_current_cost()
            logger.debug(
                f"{ith_iter}_{rnd}_{i}\t{res.get_sound_prop() * 100:.2f}%\t{res.get_unsolved_exact_prop() * 100:.2f}%\t{res.get_unsolved_edit_dis_avg():.3f}\t{res_cost:.3f}"
            )
            cost_data[i].append(res_cost)

        logger.debug(f"Used Time: {used_time:.2f}")
        # Print the current best result every K rounds
        if rnd % 250 == 100 or rnd == total_rounds - 1:
            logger.debug("Sound transformers with most exact outputs:")
            for i in range(num_programs):
                res = sound_most_exact_tfs[i][1]
                if res.is_sound():
                    logger.debug(f"{i}_{sound_most_exact_tfs[i][2]}\t{res}")
            logger.debug("Transformers with most unsolved exact outputs:")
            for i in range(num_programs):
                logger.debug(f"{i}_{most_exact_tfs[i][2]}\t{most_exact_tfs[i][1]}")
            logger.debug("Transformers with lowest cost:")
            for i in range(num_programs):
                logger.debug(f"{i}_{lowest_cost_tfs[i][2]}\t{lowest_cost_tfs[i][1]}")

    candidates_sp: list[FunctionWithCondition] = []
    candidates_p: list[FuncOp] = []
    candidates_c: list[FunctionWithCondition] = []
    if solution_size == 0:
        for i in list(sp_range) + list(p_range):
            if (
                sound_most_exact_tfs[i][1].is_sound()
                and sound_most_exact_tfs[i][1].unsolved_exacts > 0
            ):
                candidates_sp.append(
                    FunctionWithCondition(sound_most_exact_tfs[i][0].func.clone())
                )
            if (
                not most_exact_tfs[i][1].is_sound()
                and most_exact_tfs[i][1].unsolved_exacts > 0
            ):
                candidates_p.append(most_exact_tfs[i][0].func.clone())
        for i in c_range:
            if (
                sound_most_exact_tfs[i][1].is_sound()
                and sound_most_exact_tfs[i][1].unsolved_exacts > 0
            ):
                candidates_c.append(
                    FunctionWithCondition(
                        prec_set_after_distribute[i - sp_size - p_size],
                        sound_most_exact_tfs[i][0].func.clone(),
                    )
                )
    else:
        for i in range(num_programs):
            if sound_most_exact_tfs[i][1].is_sound():
                candidates_sp.append(
                    FunctionWithCondition(sound_most_exact_tfs[i][0].func.clone())
                )
            if lowest_cost_tfs[i][1].is_sound():
                candidates_sp.append(
                    FunctionWithCondition(lowest_cost_tfs[i][0].func.clone())
                )

    new_solution_set: SolutionSet = solution_set.construct_new_solution_set(
        candidates_sp, candidates_p, candidates_c, concrete_func, helper_funcs, ctx
    )

    if solution_size == 0:
        print_set_of_funcs_to_file(
            [f.to_str(eliminate_dead_code) for f in new_solution_set.solutions],
            ith_iter,
            outputs_folder,
        )

    final_cmp_res = solution_set.eval_improve([])
    logger.info(
        f"Iter {ith_iter} Finished. Exact: {final_cmp_res[0].get_exact_prop() * 100:.4f}%   Dis:{final_cmp_res[0].base_edit_dis}"
    )
    return new_solution_set


def save_solution(solution_module: ModuleOp, solution_str: str, outputs_folder: str):
    if not outputs_folder.endswith("/"):
        outputs_folder += "/"
    with open(outputs_folder + "solution.cpp", "w") as fout:
        fout.write(solution_str)
    with open(outputs_folder + "solution.mlir", "w") as fout:
        print(solution_module, file=fout)


def main() -> None:
    global ctx
    ctx = MLContext()
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(SMTDialect)
    ctx.load_dialect(SMTBitVectorDialect)
    ctx.load_dialect(SMTUtilsDialect)
    ctx.load_dialect(Transfer)
    ctx.load_dialect(Index)
    ctx.load_dialect(Comb)
    ctx.load_dialect(HW)

    # Parse the files
    module = parse_file(ctx, args.transfer_functions)
    random_number_file = args.random_file
    random_seed = args.random_seed
    num_programs = args.num_programs
    total_rounds = args.total_rounds
    program_length = args.program_length
    condition_length = args.condition_length
    inv_temp = args.inv_temp
    bitwidth = args.bitwidth
    solution_size = args.solution_size
    num_iters = args.num_iters
    weighted_dsl = args.weighted_dsl
    num_abd_procs = args.num_abd_procs
    outputs_folder = args.outputs_folder

    if num_programs is None:
        num_programs = NUM_PROGRAMS
    if total_rounds is None:
        total_rounds = TOTAL_ROUNDS
    if program_length is None:
        program_length = PROGRAM_LENGTH
    if inv_temp is None:
        inv_temp = INV_TEMP
    if bitwidth is None:
        bitwidth = SYNTH_WIDTH
    if solution_size is None:
        solution_size = SOLUTION_SIZE
    if num_iters is None:
        num_iters = NUM_ITERS
    if condition_length is None:
        condition_length = CONDITION_LENGTH
    if num_abd_procs is None:
        num_abd_procs = 0
    if outputs_folder is None:
        outputs_folder = OUTPUTS_FOLDER

    assert isinstance(module, ModuleOp)

    if not os.path.isdir(outputs_folder):
        os.mkdir(outputs_folder)

    logger = setup_loggers(outputs_folder, VERBOSE)

    logger.debug("Round_ID\tSound%\tUExact%\tUDis\tCost")

    random = Random(random_seed)
    if random_number_file is not None:
        random.read_from_file(random_number_file)

    context = SynthesizerContext(random)
    context.set_cmp_flags([0, 6, 7])
    context.use_full_int_ops()
    context.use_basic_i1_ops()

    context_weighted = SynthesizerContext(random)
    context_weighted.set_cmp_flags([0, 6, 7])
    context_weighted.use_full_int_ops()
    context_weighted.use_basic_i1_ops()

    context_cond = SynthesizerContext(random)
    context_cond.set_cmp_flags([0, 6, 7])
    context_cond.use_full_int_ops()
    context_cond.use_full_i1_ops()

    transfer_func = None

    func_name_to_func: dict[str, FuncOp] = {}
    for func in module.ops:
        if isinstance(func, FuncOp):
            func_name_to_func[func.sym_name.data] = func

    crt_func = func_name_to_func.get(CONCRETE_OP_FUNC, None)

    for func in module.ops:
        if (
            isinstance(func, FuncOp)
            and is_transfer_function(func)
            and not is_base_function(func)
        ):
            transfer_func = func
            if crt_func is None and "applied_to" in func.attributes:
                assert isa(
                    applied_to := func.attributes["applied_to"], ArrayAttr[StringAttr]
                )
                concrete_func_name = applied_to.data[0].data
                concrete_func = get_concrete_function(
                    concrete_func_name, SYNTH_WIDTH, None
                )
                crt_func = concrete_func
                break

    assert isinstance(
        transfer_func, FuncOp
    ), "No transfer function is found in input file"
    assert crt_func is not None, "Failed to get concrete function from input file"

    # Handle helper functions
    domain_constraint_func: FuncOp | None = func_name_to_func.get(
        DOMAIN_CONSTRAINT, None
    )
    instance_constraint_func: FuncOp | None = func_name_to_func.get(
        INSTANCE_CONSTRAINT, None
    )
    op_constraint_func: FuncOp | None = func_name_to_func.get(OP_CONSTRAINT, None)
    meet_func: FuncOp | None = func_name_to_func.get(MEET_FUNC, None)
    get_top_func: FuncOp | None = func_name_to_func.get(GET_TOP_FUNC, None)
    global get_top_func_op
    get_top_func_op = get_top_func
    global ret_top_func
    ret_top_func = FunctionWithCondition(construct_top_func(transfer_func))
    ret_top_func.set_func_name("ret_top")

    if meet_func is None:
        solution_size = 1

    if op_constraint_func is None:
        op_constraint_func = get_default_op_constraint(crt_func)
    assert instance_constraint_func is not None
    assert domain_constraint_func is not None
    assert meet_func is not None
    assert get_top_func is not None

    helper_funcs: list[FuncOp] = [
        crt_func,
        instance_constraint_func,
        domain_constraint_func,
        op_constraint_func,
        get_top_func,
    ]

    helper_funcs_cpp: list[str] = [print_concrete_function_to_cpp(crt_func)] + [
        print_to_cpp(func) for func in helper_funcs[1:]
    ]

    base_bodys: dict[str, FuncOp] = {}
    base_conds: dict[str, FuncOp] = {}
    base_transfers: list[FunctionWithCondition] = []
    for func in module.ops:
        if isinstance(func, FuncOp) and is_base_function(func):
            func_name = func.sym_name.data
            if func_name.endswith("_body"):
                main_name = func_name[: -len("_body")]
                if main_name in base_conds:
                    base_transfers.append(
                        FunctionWithCondition(func, base_conds.pop(main_name))
                    )
                else:
                    base_bodys[main_name] = func
            elif func_name.endswith("_cond"):
                main_name = func_name[: -len("_cond")]
                if main_name in base_bodys:
                    base_transfers.append(
                        FunctionWithCondition(base_bodys.pop(main_name), func)
                    )
                else:
                    base_conds[main_name] = func
    assert len(base_conds) == 0
    for _, func in base_bodys.items():
        base_transfers.append(FunctionWithCondition(func))

    solution_eval_func = solution_set_eval_func(
        eval_engine.AbstractDomain.KnownBits,
        bitwidth,
        helper_funcs_cpp,
    )

    if solution_size == 0:
        solution_set: SolutionSet = UnsizedSolutionSet(
            base_transfers,
            print_to_cpp,
            solution_eval_func,
            logger,
            eliminate_dead_code,
        )
    else:
        solution_set: SolutionSet = SizedSolutionSet(
            solution_size,
            base_transfers,
            print_to_cpp,
            eliminate_dead_code,
            solution_eval_func,
            logger,
        )

    eval_func = main_eval_func(
        base_transfers,
        eval_engine.AbstractDomain.KnownBits,
        bitwidth,
        helper_funcs_cpp,
    )

    # eval the initial solutions in the solution set

    init_cmp_res = solution_set.eval_improve([])
    logger.info(
        f"Initial Solution. Exact: {init_cmp_res[0].get_exact_prop() * 100:.4f}%   Dis:{init_cmp_res[0].base_edit_dis}"
    )
    print(
        f"init_solution\t{init_cmp_res[0].get_sound_prop() * 100:.4f}%\t{init_cmp_res[0].get_exact_prop() * 100:.4f}%"
    )

    # current_prog_len = 10
    # current_total_rounds = 20
    for ith_iter in range(num_iters):
        # gradually increase the program length
        # current_prog_len += (program_length - current_prog_len) // (
        #     num_iters - ith_iter
        # )
        # current_total_rounds += (total_rounds - current_total_rounds) // (
        #     num_iters - ith_iter
        # )
        print(f"Iteration {ith_iter} starts...")
        if weighted_dsl:
            assert isinstance(solution_set, UnsizedSolutionSet)
            solution_set.learn_weights(context_weighted)
        solution_set = synthesize_transfer_function(
            ith_iter,
            transfer_func,
            context,
            context_weighted,
            context_cond,
            random,
            solution_set,
            logger,
            eval_func,
            crt_func,
            helper_funcs[1:],
            ctx,
            num_programs,
            program_length,
            condition_length,
            num_abd_procs,
            total_rounds,
            solution_size,
            inv_temp,
            outputs_folder,
        )
        print(
            f"Iteration {ith_iter} finished. Size of the solution set: {solution_set.solutions_size}"
        )

        if solution_set.is_perfect:
            print("Found a perfect solution")
            break

    # Eval last solution:
    if not solution_set.has_solution():
        print("Found no solutions")
        exit(0)
    solution_module, solution_str = solution_set.generate_solution_and_cpp()
    save_solution(solution_module, solution_str, outputs_folder)
    cmp_results: list[CompareResult] = eval_engine.eval_transfer_func(
        ["solution"],
        [solution_str],
        [],
        [],
        helper_funcs_cpp + [print_to_cpp(meet_func)],
        eval_engine.AbstractDomain.KnownBits,
        bitwidth,
    )
    solution_result = cmp_results[0]
    print(
        f"last_solution\t{solution_result.get_sound_prop() * 100:.2f}%\t{solution_result.get_exact_prop() * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
