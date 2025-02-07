import argparse
import os.path
import subprocess
import time
from typing import cast

from xdsl.context import MLContext
from xdsl.parser import Parser

from io import StringIO

from xdsl.utils.hints import isa

from xdsl_smt.utils.cmp_result import CmpRes
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
from ..eval_engine.eval import eval_transfer_func
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
from xdsl.dialects.func import Func, FuncOp, Return
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

from ..utils.cost_model import compute_cost, decide
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
    returnOp = Return(combOp.results[0])
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
            returnOp = Return(combOp.results[0])
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


SYNTH_WIDTH = 8
TEST_SET_SIZE = 1000
CONCRETE_VAL_PER_TEST_CASE = 10
INSTANCE_CONSTRAINT = "getInstanceConstraint"
DOMAIN_CONSTRAINT = "getConstraint"
OP_CONSTRAINT = "op_constraint"
TMP_MODULE: list[ModuleOp] = []
ctx: MLContext

OUTPUTS_FOLDER = "outputs"


def print_func_to_file(sampler: MCMCSampler, rd: int, i: int, path: str):
    res = sampler.current_cmp
    with open(f"{path}/tf{rd}_{i}.mlir", "w") as file:
        file.write(
            f"Run: {rd}_{i}\nCost: {res.get_cost()}\nSoundness: {res.get_sound_prop()}\nPrecision: {res.get_unsolved_exact_prop()}\n"
        )
        file.write(str(sampler.get_current()))


def is_ref_function(func: FuncOp) -> bool:
    return func.sym_name.data.startswith("ref_")


# def eval_transfer_func_helper(funcs: list[FuncOp], crt_func: str, instance_constraint_func: str, domain_constraint_func: str, op_constraint_func: str, ref_func_names: list[str], ref_func_cpps: list[str]) -> list[CmpRes]:

#     cpp_codes: list[str] = []
#     func_names: list[str] = []

#     for f in funcs:
#         func_to_eval = f.clone()
#         func_names.append(f.sym_name.data)
#         cpp_code = print_to_cpp(func_to_eval)
#         cpp_codes.append(cpp_code)

#     cmp_results = eval_transfer_func(
#         func_names,
#         cpp_codes,
#         crt_func
#         + "\n"
#         + instance_constraint_func
#         + "\n"
#         + domain_constraint_func
#         + "\n"
#         + op_constraint_func,
#         ref_func_names,
#         ref_func_cpps,
#     )

#     return cmp_results


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
    possible_solution: set[str] = set()

    random = Random(random_seed)
    if random_number_file is not None:
        random.read_from_file(random_number_file)

    PROGRAM_LENGTH = 40
    NUM_PROGRAMS = 100
    INIT_COST = 1
    TOTAL_ROUNDS = 10000

    # sound_data: list[list[float]] = [[] for _ in range(NUM_PROGRAMS)]
    # precision_data: list[list[float]] = [[] for _ in range(NUM_PROGRAMS)]
    cost_data: list[list[float]] = [[] for _ in range(NUM_PROGRAMS)]

    context = SynthesizerContext(random)
    context.set_cmp_flags([0, 6, 7])

    domain_constraint_func = ""
    instance_constraint_func = ""
    op_constraint_func = get_default_op_constraint()
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

    ref_funcs: list[FuncOp] = []
    for func in module.ops:
        if isinstance(func, FuncOp) and is_ref_function(func):
            ref_funcs.append(func)

    assert len(ref_funcs) > 0
    ref_func_names = [func.sym_name.data for func in ref_funcs]
    ref_func_cpps = [print_to_cpp(func) for func in ref_funcs]

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
            func_name = func.sym_name.data
            mcmc_samplers: list[MCMCSampler] = []

            # init_code = print_to_cpp(func.clone())
            # init_soundness, init_precision = eval_transfer_func(
            #     [func_name], [init_code], crt_func
            # )

            for _ in range(NUM_PROGRAMS):
                sampler = MCMCSampler(
                    func, context, PROGRAM_LENGTH, init_cost=INIT_COST
                )
                # sampler = MCMCSampler(
                #     func, context, PROGRAM_LENGTH, init_cost=compute_cost(
                #         init_soundness[0], init_precision[0]), reset=False, init_soundness=init_soundness[0], init_precision=init_precision[0])

                mcmc_samplers.append(sampler)

            # Get the cost of initial programs
            cpp_codes: list[str] = []
            for i in range(NUM_PROGRAMS):
                func_to_eval = mcmc_samplers[i].get_current().clone()
                cpp_code = print_to_cpp(func_to_eval)
                cpp_codes.append(cpp_code)

            cmp_results: list[CmpRes] = eval_transfer_func(
                [func_name] * NUM_PROGRAMS,
                cpp_codes,
                crt_func
                + "\n"
                + instance_constraint_func
                + "\n"
                + domain_constraint_func
                + "\n"
                + op_constraint_func,
                ref_func_names,
                ref_func_cpps,
            )

            for i in range(NUM_PROGRAMS):
                mcmc_samplers[i].current_cmp = cmp_results[i]

            # MCMC start
            for round in range(TOTAL_ROUNDS):
                start = time.time()

                cpp_codes: list[str] = []
                for i in range(NUM_PROGRAMS):
                    _: float = mcmc_samplers[i].sample_next()
                    proposed_solution = mcmc_samplers[i].get_proposed()
                    assert proposed_solution is not None
                    cpp_code = print_to_cpp(proposed_solution.clone())
                    cpp_codes.append(cpp_code)

                cmp_results: list[CmpRes] = eval_transfer_func(
                    [func_name] * NUM_PROGRAMS,
                    cpp_codes,
                    crt_func
                    + "\n"
                    + instance_constraint_func
                    + "\n"
                    + domain_constraint_func
                    + "\n"
                    + op_constraint_func,
                    ref_func_names,
                    ref_func_cpps,
                )

                # num_unsound, _imprecision, num_exact, num_cases, unsolved_unsound, unsolved_imprecision, unsolved_exact, unsolved_num_cases = eval_transfer_func(
                #     [func_name] * NUM_PROGRAMS,
                #     cpp_codes,
                #     crt_func
                #     + "\n"
                #     + instance_constraint_func
                #     + "\n"
                #     + domain_constraint_func
                #     + "\n"
                #     + op_constraint_func,
                #     ref_func_names,
                #     ref_func_cpps,
                # )

                for i in range(NUM_PROGRAMS):
                    proposed_cost = cmp_results[i].get_cost()
                    current_cost = mcmc_samplers[i].current_cmp.get_cost()
                    p = random.random()
                    decision = decide(p, 200, current_cost, proposed_cost)
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

                for i in range(NUM_PROGRAMS):
                    res = mcmc_samplers[i].current_cmp
                    print(
                        f"{round}_{i}\t{res.get_sound_prop() * 100:.2f}%\t{res.get_unsolved_exact_prop() * 100:.2f}%\t{res.get_cost():.3f}"
                    )
                    # print(res)
                    cost_data[i].append(res.get_cost())

                # if soundness_percent[i] == 1 and precision_percent[i] == 1:
                #     print(mcmc_samplers[i].get_current())
                #     return
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
    for item in possible_solution:
        print(item)
