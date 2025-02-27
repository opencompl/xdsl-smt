import argparse
import os.path
import subprocess
import time
from typing import cast

from xdsl.context import MLContext
from xdsl.parser import Parser

from io import StringIO

from xdsl.utils.hints import isa

from xdsl_smt.utils.compare_result import CompareResult
from ..dialects.smt_dialect import (
    SMTDialect,
    DefineFunOp,
)
from ..dialects.smt_bitvector_dialect import (
    SMTBitVectorDialect,
    ConstantOp,
)
from xdsl_smt.dialects.transfer import (
    AbstractValueType,
    TransIntegerType,
    TupleType,
)
from ..dialects.index_dialect import Index
from ..dialects.smt_utils_dialect import SMTUtilsDialect
import xdsl_smt.eval_engine.eval as eval_engine
from xdsl.ir import BlockArgument
from xdsl.dialects.builtin import (
    Builtin,
    ModuleOp,
    IntegerAttr,
    IntegerType,
    i1,
    FunctionType,
    AnyArrayAttr,
    ArrayAttr,
    StringAttr,
)
from xdsl.dialects.func import Func, FuncOp, ReturnOp, CallOp
from ..dialects.transfer import Transfer
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.dialects.hw import HW
from ..passes.dead_code_elimination import DeadCodeElimination
from ..passes.merge_func_results import MergeFuncResultsPass
from ..passes.transfer_inline import FunctionCallInline
import xdsl.dialects.comb as comb
from xdsl.ir import Operation
from ..passes.lower_to_smt.lower_to_smt import LowerToSMTPass, SMTLowerer
from ..passes.lower_effects import LowerEffectPass
from ..passes.lower_to_smt import (
    func_to_smt_patterns,
)
from ..passes.transfer_lower import LowerToCpp
from xdsl_smt.semantics import transfer_semantics
from ..traits.smt_printer import print_to_smtlib
from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.builtin_semantics import IntegerTypeSemantics
from xdsl_smt.semantics.transfer_semantics import (
    transfer_semantics,
    AbstractValueTypeSemantics,
    TransferIntegerTypeSemantics,
)
from xdsl_smt.semantics.comb_semantics import comb_semantics
import sys as sys

from ..utils.cost_model import decide
from ..utils.mcmc_sampler import MCMCSampler
from ..utils.synthesizer_context import SynthesizerContext
from ..utils.random import Random
from ..utils.transfer_function_check_util import (
    forward_soundness_check,
    backward_soundness_check,
)
from ..utils.transfer_function_util import (
    FunctionCollection,
    SMTTransferFunction,
    fixDefiningOpReturnType,
)


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
        "-llvm_build_dir",
        type=str,
        nargs="?",
        help="Specify the build directory of LLVM",
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


def verify_pattern(ctx: MLContext, op: ModuleOp) -> bool:
    # cloned_op = op.clone()
    cloned_op = op
    stream = StringIO()
    LowerPairs().apply(ctx, cloned_op)
    CanonicalizePass().apply(ctx, cloned_op)
    DeadCodeElimination().apply(ctx, cloned_op)

    print_to_smtlib(cloned_op, stream)
    res = subprocess.run(
        ["z3", "-in"],
        capture_output=True,
        input=stream.getvalue(),
        text=True,
    )
    if res.returncode != 0:
        raise Exception(res.stderr)
    return "unsat" in res.stdout


def get_model(ctx: MLContext, op: ModuleOp) -> tuple[bool, str]:
    cloned_op = op.clone()
    stream = StringIO()
    LowerPairs().apply(ctx, cloned_op)
    CanonicalizePass().apply(ctx, cloned_op)
    DeadCodeElimination().apply(ctx, cloned_op)

    print_to_smtlib(cloned_op, stream)
    print("\n(eval const_first)\n", file=stream)
    # print(stream.getvalue())
    res = subprocess.run(
        ["z3", "-in"],
        capture_output=True,
        input=stream.getvalue(),
        text=True,
    )
    if res.returncode != 0:
        return False, ""
    return True, str(res.stdout)


def lowerToSMTModule(module: ModuleOp, width: int, ctx: MLContext):
    # lower to SMT
    SMTLowerer.rewrite_patterns = {
        **func_to_smt_patterns,
    }
    SMTLowerer.type_lowerers = {
        IntegerType: IntegerTypeSemantics(),
        AbstractValueType: AbstractValueTypeSemantics(),
        TransIntegerType: TransferIntegerTypeSemantics(width),
        # tuple and abstract use the same type lowerers
        TupleType: AbstractValueTypeSemantics(),
    }
    SMTLowerer.op_semantics = {
        **arith_semantics,
        **transfer_semantics,
        **comb_semantics,
    }
    LowerToSMTPass().apply(ctx, module)
    MergeFuncResultsPass().apply(ctx, module)
    LowerEffectPass().apply(ctx, module)


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


def need_replace_int_attr(func: FuncOp) -> bool:
    if "replace_int_attr" in func.attributes:
        forward = func.attributes["replace_int_attr"]
        assert isinstance(forward, IntegerAttr)
        return forward.value.data == 1
    return False


def get_operationNo(func: FuncOp) -> int:
    if "operationNo" in func.attributes:
        assert isinstance(func.attributes["operationNo"], IntegerAttr)
        return func.attributes["operationNo"].value.data
    return -1


def get_int_attr_arg(func: FuncOp) -> list[int]:
    int_attr: list[int] = []
    assert "int_attr" in func.attributes
    func_int_attr = func.attributes["int_attr"]
    assert isa(func_int_attr, AnyArrayAttr)
    for attr in func_int_attr.data:
        assert isinstance(attr, IntegerAttr)
        int_attr.append(attr.value.data)
    return int_attr


def generateIntAttrArg(int_attr_arg: list[int] | None) -> dict[int, int]:
    if int_attr_arg is None:
        return {}
    intAttr: dict[int, int] = {}
    for i in int_attr_arg:
        intAttr[i] = 0
    return intAttr


def nextIntAttrArg(intAttr: dict[int, int], width: int) -> bool:
    if not intAttr:
        return False
    maxArity: int = 0
    for i in intAttr.keys():
        maxArity = max(i, maxArity)
    hasCarry: bool = True
    for i in range(maxArity, -1, -1):
        if not hasCarry:
            break
        if i in intAttr:
            intAttr[i] += 1
            if intAttr[i] >= width:
                intAttr[i] %= width
            else:
                hasCarry = False
    return not hasCarry


def create_smt_function(func: FuncOp, width: int, ctx: MLContext) -> DefineFunOp:
    global TMP_MODULE
    TMP_MODULE.append(ModuleOp([func.clone()]))
    lowerToSMTModule(TMP_MODULE[-1], width, ctx)
    resultFunc = TMP_MODULE[-1].ops.first
    assert isinstance(resultFunc, DefineFunOp)
    return resultFunc


def get_dynamic_transfer_function(
    func: FuncOp, width: int, module: ModuleOp, int_attr: dict[int, int], ctx: MLContext
) -> DefineFunOp:
    module.body.block.add_op(func)
    args: list[BlockArgument] = []
    for arg_idx, val in int_attr.items():
        bv_constant = ConstantOp(val, width)
        assert isinstance(func.body.block.first_op, Operation)
        func.body.block.insert_op_before(bv_constant, func.body.block.first_op)
        args.append(func.body.block.args[arg_idx])
        args[-1].replace_by(bv_constant.res)
    for arg in args:
        func.body.block.erase_arg(arg)
    new_args_type = [arg.type for arg in func.body.block.args]
    new_function_type = FunctionType.from_lists(
        new_args_type, func.function_type.outputs.data
    )
    func.function_type = new_function_type

    lowerToSMTModule(module, width, ctx)
    resultFunc = module.ops.first
    assert isinstance(resultFunc, DefineFunOp)
    return fixDefiningOpReturnType(resultFunc)


def get_dynamic_concrete_function_name(concrete_op_name: str) -> str:
    if concrete_op_name == "comb.extract":
        return "comb_extract"
    assert False and "Unsupported concrete function"


# Used to construct concrete operations with integer attrs when enumerating all possible int attrs
# Thus this can only be constructed at the run time
def get_dynamic_concrete_function(
    concrete_func_name: str, width: int, intAttr: dict[int, int], is_forward: bool
) -> FuncOp:
    result = None
    intTy = IntegerType(width)
    combOp = None
    if concrete_func_name == "comb_extract":
        delta: int = 1 if not is_forward else 0
        resultWidth = intAttr[1 + delta]
        resultIntTy = IntegerType(resultWidth)
        low_bit = intAttr[2 + delta]
        funcTy = FunctionType.from_lists([intTy], [resultIntTy])
        result = FuncOp(concrete_func_name, funcTy)
        combOp = comb.ExtractOp(
            result.args[0], IntegerAttr.from_int_and_width(low_bit, 64), resultIntTy
        )
    else:
        print(concrete_func_name)
        assert False and "Not supported concrete function yet"
    returnOp = ReturnOp(combOp.results[0])
    result.body.block.add_ops([combOp, returnOp])
    assert result is not None and (
        "Cannot find the concrete function for" + concrete_func_name
    )
    return result


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


def soundness_check(
    smt_transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    int_attr: dict[int, int],
    ctx: MLContext,
):
    query_module = ModuleOp([])
    if smt_transfer_function.is_forward:
        added_ops: list[Operation] = forward_soundness_check(
            smt_transfer_function,
            domain_constraint,
            instance_constraint,
            int_attr,
        )
    else:
        added_ops: list[Operation] = backward_soundness_check(
            smt_transfer_function,
            domain_constraint,
            instance_constraint,
            int_attr,
        )
    query_module.body.block.add_ops(added_ops)
    FunctionCallInline(True, {}).apply(ctx, query_module)
    verify_res = verify_pattern(ctx, query_module)
    print("Soundness Check result:", verify_res)
    return verify_res


def print_concrete_function_to_cpp(func: FuncOp) -> str:
    sio = StringIO()
    LowerToCpp(sio, True).apply(ctx, cast(ModuleOp, func))
    return sio.getvalue()


def print_to_cpp(func: FuncOp) -> str:
    sio = StringIO()
    LowerToCpp(sio).apply(ctx, cast(ModuleOp, func))
    return sio.getvalue()


def get_default_op_constraint():
    return """
    int op_constraint(APInt arg0,APInt arg1){
	return true;
    }
    """


SYNTH_WIDTH = 4
TEST_SET_SIZE = 1000
CONCRETE_VAL_PER_TEST_CASE = 10
PROGRAM_LENGTH = 40
NUM_PROGRAMS = 100
INIT_COST = 1
TOTAL_ROUNDS = 10000
INV_TEMP = 200
SOLUTION_SIZE = 8
INSTANCE_CONSTRAINT = "getInstanceConstraint"
DOMAIN_CONSTRAINT = "getConstraint"
OP_CONSTRAINT = "op_constraint"
MEET_FUNC = "meet"
TMP_MODULE: list[ModuleOp] = []
ctx: MLContext

OUTPUTS_FOLDER = "outputs"


def print_func_to_file(sampler: MCMCSampler, rd: int, i: int, path: str):
    res = sampler.current_cmp
    with open(f"{path}/tf{rd}_{i}.mlir", "w") as file:
        file.write(
            f"Run: {rd}_{i}\nCost: {res.get_cost()}\nSound: {res.get_sound_prop()}\nUExact: {res.get_unsolved_exact_prop()}\nUDis: {res.get_unsolved_edit_dis_avg()}\n{res}\n"
        )
        file.write(str(sampler.get_current()))


def is_ref_function(func: FuncOp) -> bool:
    return func.sym_name.data.startswith("ref_")


def generate_meet_solution(meet_funcs: list[FuncOp]) -> FuncOp:
    if len(meet_funcs) == 0:
        raise ValueError("Solution candidate not found!")
    result = FuncOp("solution", meet_funcs[0].function_type)
    result_type = result.function_type.outputs.data
    part_result: list[CallOp] = []
    for ith, func in enumerate(meet_funcs):
        cur_func_name = "part_solution_" + str(ith)
        func.sym_name = StringAttr("part_solution_" + str(ith))
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
    inv_temp = args.inv_temp
    bitwidth = args.bitwidth
    solution_size = args.solution_size

    # Set up llvm_build_dir
    llvm_build_dir = args.llvm_build_dir
    if llvm_build_dir is not None:
        if not llvm_build_dir.endswith("/"):
            llvm_build_dir += "/"
        llvm_build_dir += "bin/"
        eval_engine.llvm_bin_dir = llvm_build_dir
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

    assert isinstance(module, ModuleOp)

    if not os.path.isdir(OUTPUTS_FOLDER):
        os.mkdir(OUTPUTS_FOLDER)

    """
    (
        smt_transfer_function_obj,
        domain_constraint,
        instance_constraint,
        int_attr,
    ) = get_transfer_function(module, ctx)
    test_set = generate_test_set(
        smt_transfer_function_obj, domain_constraint, instance_constraint, int_attr, ctx
    )
    """
    print("Round\tsoundness%\tprecision%\tcost")
    """
    This structure maintains a dictionary of potential solutions
    All solutions must come from different program groups, sorted by cost
    """
    solutions: dict[int, tuple[float, FuncOp]] = {}

    random = Random(random_seed)
    if random_number_file is not None:
        random.read_from_file(random_number_file)

    cost_data: list[list[float]] = [[] for _ in range(num_programs)]

    context = SynthesizerContext(random)
    context.set_cmp_flags([0, 6, 7])
    context.use_full_int_ops()

    domain_constraint_func = ""
    instance_constraint_func = ""
    op_constraint_func = get_default_op_constraint()
    meet_func = ""
    # Handle helper funcitons
    for func in module.ops:
        if isinstance(func, FuncOp):
            func_name = func.sym_name.data
            if func_name == DOMAIN_CONSTRAINT:
                domain_constraint_func = print_to_cpp(func)
            elif func_name == INSTANCE_CONSTRAINT:
                instance_constraint_func = print_to_cpp(func)
            elif func_name == OP_CONSTRAINT:
                op_constraint_func = print_to_cpp(func)
            elif func_name == MEET_FUNC:
                meet_func = print_to_cpp(func)
    if meet_func == "":
        solution_size = 1

    ref_funcs: list[FuncOp] = []
    for func in module.ops:
        if isinstance(func, FuncOp) and is_ref_function(func):
            ref_funcs.append(func)

    # We comment this out because ref functions could be empty
    # assert len(ref_funcs) > 0
    ref_func_names = [func.sym_name.data for func in ref_funcs]
    ref_func_cpps = [print_to_cpp(func) for func in ref_funcs]
    used_crt_func = None

    for func in module.ops:
        if isinstance(func, FuncOp) and is_transfer_function(func):
            concrete_func_name = ""
            if isinstance(func, FuncOp) and "applied_to" in func.attributes:
                assert isa(
                    applied_to := func.attributes["applied_to"], ArrayAttr[StringAttr]
                )
                concrete_func_name = applied_to.data[0].data
            concrete_func = get_concrete_function(concrete_func_name, SYNTH_WIDTH, None)
            crt_func = print_concrete_function_to_cpp(concrete_func)
            if used_crt_func is None:
                used_crt_func = crt_func
            func_name = func.sym_name.data
            mcmc_samplers: list[MCMCSampler] = []

            for _ in range(num_programs):
                sampler = MCMCSampler(
                    func,
                    context,
                    program_length,
                    random_init_program=True,
                    init_cost=INIT_COST,
                )
                mcmc_samplers.append(sampler)

            # Get the cost of initial programs
            cpp_codes: list[str] = []
            for i in range(num_programs):
                func_to_eval = mcmc_samplers[i].get_current().clone()
                cpp_code = print_to_cpp(func_to_eval)
                cpp_codes.append(cpp_code)

            cmp_results: list[CompareResult] = eval_engine.eval_transfer_func(
                [func_name] * num_programs,
                cpp_codes,
                crt_func,
                ref_func_names,
                ref_func_cpps,
                eval_engine.AbstractDomain.KnownBits,
                bitwidth,
                [instance_constraint_func, domain_constraint_func, op_constraint_func],
            )

            for i in range(num_programs):
                mcmc_samplers[i].current_cmp = cmp_results[i]

            # MCMC start
            for round in range(total_rounds):
                start = time.time()

                cpp_codes: list[str] = []
                for i in range(num_programs):
                    _: float = mcmc_samplers[i].sample_next()
                    proposed_solution = mcmc_samplers[i].get_proposed()

                    assert proposed_solution is not None
                    cpp_code = print_to_cpp(proposed_solution.clone())
                    cpp_codes.append(cpp_code)

                cmp_results: list[CompareResult] = eval_engine.eval_transfer_func(
                    [func_name] * num_programs,
                    cpp_codes,
                    crt_func,
                    ref_func_names,
                    ref_func_cpps,
                    eval_engine.AbstractDomain.KnownBits,
                    bitwidth,
                    [
                        instance_constraint_func,
                        domain_constraint_func,
                        op_constraint_func,
                    ],
                )

                for i in range(num_programs):
                    proposed_cost = cmp_results[i].get_cost()
                    current_cost = mcmc_samplers[i].current_cmp.get_cost()
                    p = random.random()
                    decision = decide(p, inv_temp, current_cost, proposed_cost)
                    if decision:
                        cost_reduce = current_cost - proposed_cost

                        mcmc_samplers[i].accept_proposed(cmp_results[i])

                        assert mcmc_samplers[i].get_proposed() is None

                        if cost_reduce > 0:
                            print_func_to_file(
                                mcmc_samplers[i], round, i, OUTPUTS_FOLDER
                            )

                    else:
                        mcmc_samplers[i].reject_proposed()
                        pass

                for i in range(num_programs):
                    res = mcmc_samplers[i].current_cmp
                    print(
                        f"{round}_{i}\t{res.get_sound_prop() * 100:.2f}%\t{res.get_unsolved_exact_prop() * 100:.2f}%\t{res.get_unsolved_edit_dis_avg():.3f}\t{res.get_cost():.3f}"
                    )
                    cost_data[i].append(res.get_cost())
                    """
                    Select solution candidates
                    """
                    if res.sounds == res.all_cases:
                        cost = res.get_cost()
                        # If solutions has less than eight element, we directly add it
                        if len(solutions) < solution_size:
                            if i in solutions:
                                if cost < solutions[i][0]:
                                    solutions[i] = (
                                        cost,
                                        mcmc_samplers[i].current.func.clone(),
                                    )
                            else:
                                solutions[i] = (
                                    cost,
                                    mcmc_samplers[i].current.func.clone(),
                                )
                        # Otherwise we replace the one with maximal cost in current solution
                        else:
                            max_ith = -1
                            max_cost = 0
                            if i in solutions:
                                max_ith = i
                                max_cost = solutions[i][0]
                            else:
                                for ith, ith_ele in solutions.items():
                                    if ith_ele[0] > max_cost:
                                        max_ith = ith
                                        max_cost = ith_ele[0]
                            if cost < max_cost:
                                del solutions[max_ith]
                                solutions[i] = (
                                    cost,
                                    mcmc_samplers[i].current.func.clone(),
                                )
                        assert len(solutions) <= solution_size
                end = time.time()
                used_time = end - start

                print(f"Used Time: {used_time:.2f}")
                """
                tmp_clone_module: ModuleOp = module.clone()


                lowerToSMTModule(tmp_clone_module, SYNTH_WIDTH, ctx)
                for smt_func in tmp_clone_module.ops:
                    if (
                        isinstance(smt_func, DefineFunOp)
                        and smt_func.fun_name.data == func_name
                    ):
                        smt_transfer_function_obj.transfer_function = smt_func

                        soundness_check_res = soundness_check(
                            smt_transfer_function_obj,
                            domain_constraint,
                            instance_constraint,
                            int_attr,
                            ctx,
                        )
                        if soundness_check_res:
                            print(mcmcSampler.func)
                        """
    # Eval last solution:
    solution_str = ""
    solution_list: list[FuncOp] = []
    if len(solutions) == 0:
        print("Cannot find any sound transfer functions!")
        exit(0)
    for item in solutions.values():
        solution_list.append(item[1])
    last_solution = generate_meet_solution(solution_list)
    for sol in solution_list:
        solution_str += print_to_cpp(sol)
        solution_str += "\n"
    solution_str += print_to_cpp(last_solution)
    solution_str += "\n"
    with open("tmp.cpp", "w") as fout:
        fout.write(solution_str)
    assert used_crt_func is not None
    cmp_results: list[CompareResult] = eval_engine.eval_transfer_func(
        ["solution"],
        [solution_str],
        used_crt_func,
        ref_func_names,
        ref_func_cpps,
        eval_engine.AbstractDomain.KnownBits,
        bitwidth,
        [
            instance_constraint_func,
            domain_constraint_func,
            op_constraint_func,
            meet_func,
        ],
    )
    solution_result = cmp_results[0]
    print(
        f"last_solution\t{solution_result.get_sound_prop() * 100:.2f}%\t{solution_result.get_exact_prop() * 100:.2f}%\t{solution_result.get_unsolved_edit_dis_avg():.3f}\t{solution_result.get_cost():.3f}"
    )
