import argparse
import subprocess

from xdsl.context import MLContext
from xdsl.parser import Parser

from io import StringIO

from xdsl.utils.hints import isa
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
from xdsl.ir.core import BlockArgument
from xdsl.dialects.builtin import (
    Builtin,
    ModuleOp,
    IntegerAttr,
    IntegerType,
    i1,
    FunctionType,
    ArrayAttr,
    StringAttr,
    AnyArrayAttr,
)
from xdsl.dialects.func import Func, FuncOp, Return
from ..dialects.transfer import Transfer
from xdsl.dialects.arith import Arith
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
from ..utils.transfer_function_util import (
    SMTTransferFunction,
    FunctionCollection,
    TransferFunction,
    fixDefiningOpReturnType,
)

from ..utils.transfer_function_check_util import (
    forward_soundness_check,
    backward_soundness_check,
    counterexample_check,
    int_attr_check,
)
from ..passes.transfer_unroll_loop import UnrollTransferLoop
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


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "transfer_functions", type=str, nargs="?", help="path to the transfer functions"
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


def solveVectorWidth():
    return list(range(4, 5))


def verify_pattern(ctx: MLContext, op: ModuleOp) -> bool:
    cloned_op = op.clone()
    stream = StringIO()
    LowerPairs().apply(ctx, cloned_op)
    CanonicalizePass().apply(ctx, cloned_op)
    DeadCodeElimination().apply(ctx, cloned_op)

    print_to_smtlib(cloned_op, stream)
    # print(stream.getvalue())
    res = subprocess.run(
        ["z3", "-in"],
        capture_output=True,
        input=stream.getvalue(),
        text=True,
    )
    if res.returncode != 0:
        raise Exception(res.stderr)
    return "unsat" in res.stdout


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


# Given a name of one concrete operation, return a function with only that operation
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
            func_name = concrete_op_name.replace(".", "_")

            if concrete_op_name == "comb.mux":
                funcTy = FunctionType.from_lists([i1, intTy, intTy], [intTy])
                result = FuncOp(func_name, funcTy)
                combOp = k(*result.args)
            elif concrete_op_name == "comb.icmp":
                funcTy = FunctionType.from_lists([intTy, intTy], [i1])
                result = FuncOp(func_name, funcTy)
                assert extra is not None
                func_name += str(extra)
                combOp = comb.ICmpOp(result.args[0], result.args[1], extra)
            elif concrete_op_name == "comb.concat":
                funcTy = FunctionType.from_lists(
                    [intTy, intTy], [IntegerType(width * 2)]
                )
                result = FuncOp(func_name, funcTy)
                combOp = comb.ConcatOp.from_int_values(result.args)
            else:
                funcTy = FunctionType.from_lists([intTy, intTy], [intTy])
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


KEY_NEED_VERIFY = "builtin.NEED_VERIFY"
MAXIMAL_VERIFIED_BITS = 8
INSTANCE_CONSTRAINT = "getInstanceConstraint"
DOMAIN_CONSTRAINT = "getConstraint"
TMP_MODULE: list[ModuleOp] = []
ctx: MLContext


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


def verify_smt_transfer_function(
    smt_transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    width: int,
    transfer_function: TransferFunction,
    func_name_to_func: dict[str, FuncOp],
    dynamic_transfer_functions: dict[str, FuncOp],
    ctx: MLContext,
) -> None:
    # Soundness check
    query_module = ModuleOp([])
    dynamic_concrete_function_module = ModuleOp([])
    dynamic_transfer_function_module = ModuleOp([])
    int_attr = generateIntAttrArg(smt_transfer_function.int_attr_arg)

    # enumerating all possible int attr
    while True:
        query_module = ModuleOp([])
        added_ops: list[Operation] = int_attr_check(
            smt_transfer_function, domain_constraint, instance_constraint, int_attr
        )
        query_module.body.block.add_ops(added_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)
        # LowerToSMTPass().apply(ctx, query_module)
        # print(int_attr)
        # we find int_attr is satisfiable
        if not verify_pattern(ctx, query_module):
            # print(int_attr)
            # start to create dynamic concrete function
            # and update smt_transfer_function
            if smt_transfer_function.concrete_function is None:
                dynamic_concrete_function_module = ModuleOp([])
                dynamic_concrete_function_module.body.block.add_ops(
                    [
                        get_dynamic_concrete_function(
                            smt_transfer_function.concrete_function_name,
                            width,
                            int_attr,
                            transfer_function.is_forward,
                        )
                    ]
                )
                lowerToSMTModule(dynamic_concrete_function_module, width, ctx)
                assert len(dynamic_concrete_function_module.ops) == 1
                assert isinstance(
                    dynamic_concrete_function_module.ops.first, DefineFunOp
                )
                concrete_func = dynamic_concrete_function_module.ops.first
                smt_transfer_function.concrete_function = concrete_func

            assert smt_transfer_function.concrete_function is not None

            if smt_transfer_function.transfer_function is None:
                # dynamic create transfer_function with given int_attr
                dynamic_transfer_function_module = ModuleOp([])
                smt_transfer_function.transfer_function = get_dynamic_transfer_function(
                    dynamic_transfer_functions[
                        smt_transfer_function.transfer_function_name
                    ],
                    width,
                    dynamic_transfer_function_module,
                    int_attr,
                    ctx,
                )

            assert smt_transfer_function.transfer_function is not None

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
            # print(query_module)
            # LowerToSMTPass().apply(ctx, query_module)
            # print_to_smtlib(query_module, sys.stdout)

            print("Soundness Check result:", verify_pattern(ctx, query_module))

            if smt_transfer_function.soundness_counterexample is not None:
                query_module = ModuleOp([])
                soundness_counterexample_func_name = (
                    smt_transfer_function.soundness_counterexample.fun_name
                )
                assert soundness_counterexample_func_name is not None
                counterexample_func_name = soundness_counterexample_func_name.data
                added_ops = counterexample_check(
                    func_name_to_func[counterexample_func_name],
                    smt_transfer_function.soundness_counterexample,
                    domain_constraint,
                    int_attr,
                )
                query_module.body.block.add_ops(added_ops)
                FunctionCallInline(True, {}).apply(ctx, query_module)
                # LowerToSMTPass().apply(ctx, query_module)

                print(
                    "Unable to find soundness counterexample: ",
                    verify_pattern(ctx, query_module),
                )

        hasNext = nextIntAttrArg(int_attr, width)
        if not hasNext:
            break


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

    # Parse the files
    module = parse_file(ctx, args.transfer_functions)
    assert isinstance(module, ModuleOp)

    func_name_to_func: dict[str, FuncOp] = {}
    transfer_functions: dict[str, TransferFunction] = {}
    domain_constraint: FunctionCollection | None = None
    instance_constraint: FunctionCollection | None = None
    for func in module.ops:
        if isinstance(func, FuncOp):
            func_name = func.sym_name.data
            func_name_to_func[func_name] = func

            # Check func validity
            assert len(func.function_type.inputs) == len(func.args)
            for func_type_arg, arg in zip(func.function_type.inputs, func.args):
                assert func_type_arg == arg.type
            return_op = func.body.block.last_op
            assert return_op is not None and isinstance(return_op, Return)
            assert return_op.operands[0].type == func.function_type.outputs.data[0]
            # End of check function type

            if is_transfer_function(func):
                transfer_functions[func_name] = TransferFunction(
                    func,
                    is_forward(func),
                    get_operationNo(func),
                    need_replace_int_attr(func),
                )
            if func_name == DOMAIN_CONSTRAINT:
                assert domain_constraint is None
                domain_constraint = FunctionCollection(func, create_smt_function, ctx)
            elif func_name == INSTANCE_CONSTRAINT:
                assert instance_constraint is None
                instance_constraint = FunctionCollection(func, create_smt_function, ctx)

    assert domain_constraint is not None
    assert instance_constraint is not None

    FunctionCallInline(False, func_name_to_func).apply(ctx, module)

    for width in solveVectorWidth():
        print("Current width: ", width)
        smt_module = module.clone()

        # expand for loops
        unrollTransferLoop = UnrollTransferLoop(width)
        assert isinstance(smt_module, ModuleOp)
        unrollTransferLoop.apply(ctx, smt_module)
        concrete_funcs: list[FuncOp] = []
        concrete_func_names: set[str] = set()
        transfer_function_name_to_concrete_function_name: dict[str, str] = {}
        dynamic_transfer_functions: dict[str, FuncOp] = {}

        # add concrete operations for every transfer functions
        for op in smt_module.ops:
            # op is a transfer function
            if isinstance(op, FuncOp) and "applied_to" in op.attributes:
                assert isa(
                    applied_to := op.attributes["applied_to"], ArrayAttr[StringAttr]
                )
                concrete_func_name = applied_to.data[0].data
                # concrete_func_name = op.attributes["applied_to"].data[0].data
                func_name = op.sym_name.data

                if need_replace_int_attr(op):
                    func_name = op.sym_name.data
                    dynamic_transfer_functions[func_name] = op
                    op.detach()

                if "int_attr" in op.attributes:
                    transfer_function_name_to_concrete_function_name[
                        func_name
                    ] = get_dynamic_concrete_function_name(concrete_func_name)
                    continue

                if concrete_func_name not in concrete_func_names:
                    extra = None
                    assert isa(
                        applied_to := op.attributes["applied_to"], ArrayAttr[StringAttr]
                    )
                    if len(applied_to.data) > 1:
                        extra = applied_to.data[1]
                        assert (
                            isinstance(extra, IntegerAttr)
                            and "only support for integer attr for the second applied arg for now"
                        )
                        extra = extra.value.data
                    concrete_funcs.append(
                        get_concrete_function(concrete_func_name, width, extra)
                    )
                    concrete_func_names.add(concrete_func_name)
                transfer_function_name_to_concrete_function_name[
                    func_name
                ] = concrete_funcs[-1].sym_name.data

        smt_module.body.block.add_ops(concrete_funcs)
        lowerToSMTModule(smt_module, width, ctx)

        func_name_to_smt_func: dict[str, DefineFunOp] = {}
        for op in smt_module.ops:
            if isinstance(op, DefineFunOp):
                op_func_name = op.fun_name
                assert op_func_name is not None
                func_name = op_func_name.data
                func_name_to_smt_func[func_name] = op

        for func_name, transfer_function in transfer_functions.items():
            concrete_func_name = transfer_function_name_to_concrete_function_name[
                func_name
            ]
            concrete_func = None
            if concrete_func_name in func_name_to_smt_func:
                concrete_func = func_name_to_smt_func[concrete_func_name]

            smt_transfer_function = None
            if func_name in func_name_to_smt_func:
                smt_transfer_function = func_name_to_smt_func[func_name]

            print("Current verify: ", func_name)
            abs_op_constraint = None
            if "abs_op_constraint" in transfer_function.transfer_function.attributes:
                abs_op_constraint_name_attr = (
                    transfer_function.transfer_function.attributes["abs_op_constraint"]
                )
                assert isinstance(abs_op_constraint_name_attr, StringAttr)
                abs_op_constraint_name = abs_op_constraint_name_attr.data
                abs_op_constraint = func_name_to_smt_func[abs_op_constraint_name]

            op_constraint = None
            if "op_constraint" in transfer_function.transfer_function.attributes:
                op_constraint_func_name_attr = (
                    transfer_function.transfer_function.attributes["op_constraint"]
                )
                assert isinstance(op_constraint_func_name_attr, StringAttr)
                op_constraint_func_name = op_constraint_func_name_attr.data
                op_constraint = func_name_to_smt_func[op_constraint_func_name]

            soundness_counterexample = None
            if (
                "soundness_counterexample"
                in transfer_function.transfer_function.attributes
            ):
                soundness_counterexample_func_name_attr = (
                    transfer_function.transfer_function.attributes[
                        "soundness_counterexample"
                    ]
                )
                assert isinstance(soundness_counterexample_func_name_attr, StringAttr)
                soundness_counterexample_func_name = (
                    soundness_counterexample_func_name_attr.data
                )
                soundness_counterexample = func_name_to_smt_func[
                    soundness_counterexample_func_name
                ]

            int_attr_arg = None
            int_attr_constraint = None
            if "int_attr" in transfer_function.transfer_function.attributes:
                assert (
                    "int_attr_constraint"
                    in transfer_function.transfer_function.attributes
                )
                int_attr_arg = get_int_attr_arg(transfer_function.transfer_function)
                int_attr_constraint_name_attr = (
                    transfer_function.transfer_function.attributes[
                        "int_attr_constraint"
                    ]
                )
                assert isinstance(int_attr_constraint_name_attr, StringAttr)
                int_attr_constraint_name = int_attr_constraint_name_attr.data
                int_attr_constraint = func_name_to_smt_func[int_attr_constraint_name]

            smt_transfer_function_obj = SMTTransferFunction(
                func_name,
                smt_transfer_function,
                transfer_functions,
                concrete_func_name,
                concrete_func,
                abs_op_constraint,
                op_constraint,
                soundness_counterexample,
                int_attr_arg,
                int_attr_constraint,
            )
            verify_smt_transfer_function(
                smt_transfer_function_obj,
                domain_constraint,
                instance_constraint,
                width,
                transfer_function,
                func_name_to_func,
                dynamic_transfer_functions,
                ctx,
            )
