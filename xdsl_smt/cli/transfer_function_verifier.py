import argparse
import subprocess

from xdsl.ir import MLContext, Attribute
from xdsl.parser import Parser

from io import StringIO

from ..dialects.smt_dialect import (
    SMTDialect,
    DefineFunOp,
    DeclareConstOp,
    CallOp,
    AssertOp,
    CheckSatOp,
    EqOp,
    ConstantBoolOp,
    ImpliesOp,
    ForallOp,
    AndOp,
    YieldOp,
    BoolType,
)
from ..dialects.smt_bitvector_dialect import (
    SMTBitVectorDialect,
    ConstantOp,
    BitVectorType,
    ExtractOp,
)
from ..dialects.smt_utils_dialect import FirstOp, PairOp, PairType
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
    Region,
    Block,
)
from xdsl.dialects.func import Func, FuncOp, Return, Call
from ..dialects.transfer import Transfer
from xdsl.dialects.arith import Arith
from ..passes.transfer_inline import FunctionCallInline
import xdsl.dialects.comb as comb
from xdsl.ir import Operation
from ..passes.lower_to_smt.lower_to_smt import LowerToSMT, integer_poison_type_lowerer
from ..passes.lower_to_smt import (
    func_to_smt_patterns,
)
from ..utils.transfer_function_util import (
    getArgumentInstances,
    getResultInstance,
    callFunctionAndAssertResult,
    callFunction,
    getArgumentWidths,
    getResultWidth,
    compress_and_op,
    SMTTransferFunction,
    FunctionCollection,
    TransferFunction,
)

from ..utils.transfer_function_check_util import (
    forward_soundness_check,
    backward_soundness_check,
    valid_abstract_domain_check,
    counterexample_check,
    int_attr_check,
)
from ..passes.transfer_unroll_loop import UnrollTransferLoop
from xdsl_smt.semantics import transfer_semantics
from ..traits.smt_printer import print_to_smtlib
from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl_smt.passes.canonicalize_smt import CanonicalizeSMT
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.transfer_semantics import (
    transfer_semantics,
    abstract_value_type_lowerer,
    transfer_integer_type_lowerer,
)
from xdsl_smt.semantics.comb_semantics import comb_semantics
import sys as sys


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "transfer_functions", type=str, nargs="?", help="path to the transfer functions"
    )


def parse_file(ctx, file: str | None) -> Operation:
    if file is None:
        f = sys.stdin
    else:
        f = open(file)

    parser = Parser(ctx, f.read(), file)
    module = parser.parse_op()
    return module


def solveVectorWidth():
    return list(range(4, 5))


def verify_pattern(ctx: MLContext, op: ModuleOp) -> bool:
    cloned_op = op.clone()
    LowerPairs().apply(ctx, cloned_op)
    CanonicalizeSMT().apply(ctx, cloned_op)
    stream = StringIO()
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
    funcTy: FunctionType = None
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
            funcTy: FunctionType = None
            func_name = concrete_op_name.replace(".", "_")
            combOp = None

            if concrete_op_name == "comb.mux":
                funcTy = FunctionType.from_lists([i1, intTy, intTy], [intTy])
                result = FuncOp(func_name, funcTy)
                combOp = k(*result.args)
            elif concrete_op_name == "comb.icmp":
                funcTy = FunctionType.from_lists([intTy, intTy], [i1])
                result = FuncOp(func_name, funcTy)
                func_name += str(extra)
                combOp = comb.ICmpOp(result.args[0], result.args[1], extra)
            else:
                funcTy = FunctionType.from_lists([intTy, intTy], [intTy])
                result = FuncOp(func_name, funcTy)
                if (
                    issubclass(k, comb.VariadicCombOperation)
                    or concrete_op_name == "comb.concat"
                ):
                    combOp = k.create(operands=result.args, result_types=[intTy])
                else:
                    combOp = k(*result.args)

            returnOp = Return(combOp.results[0])
            result.body.block.add_ops([combOp, returnOp])
    assert result is not None and (
        "Cannot find the concrete function for" + concrete_op_name
    )
    return result


def lowerToSMTModule(module, width, ctx):
    # lower to SMT
    LowerToSMT.rewrite_patterns = [
        *func_to_smt_patterns,
    ]
    LowerToSMT.type_lowerers = [
        integer_poison_type_lowerer,
        abstract_value_type_lowerer,
        lambda type: transfer_integer_type_lowerer(type, width),
    ]
    LowerToSMT.operation_semantics = {
        **arith_semantics,
        **transfer_semantics,
        **comb_semantics,
    }
    LowerToSMT().apply(ctx, module)


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
    int_attr = []
    for attr in func.attributes["int_attr"].data:
        assert isinstance(attr, IntegerAttr)
        int_attr.append(attr.value.data)
    return int_attr


def generateIntAttrArg(int_attr_arg: list[int] | None) -> dict[int, int]:
    if int_attr_arg is None:
        return {}
    intAttr = {}
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
ctx: MLContext = None


def create_smt_function(func: FuncOp, width: int) -> DefineFunOp:
    global TMP_MODULE
    TMP_MODULE.append(ModuleOp([func.clone()]))
    lowerToSMTModule(TMP_MODULE[-1], width, ctx)
    resultFunc = TMP_MODULE[-1].ops.first
    assert isinstance(resultFunc, DefineFunOp)
    return resultFunc


def get_dynamic_transfer_function(
    func: FuncOp, width: int, module: ModuleOp, int_attr: dict[int, int]
) -> DefineFunOp:
    module.body.block.add_op(func)
    args: list[BlockArgument] = []
    for arg_idx, val in int_attr.items():
        bv_constant = ConstantOp(val, width)
        func.body.block.insert_op_before(bv_constant, func.body.block.first_op)
        args.append(func.body.block.args[arg_idx])
        args[-1].replace_by(bv_constant.res)
    for arg in args:
        func.body.block.erase_arg(arg)
    new_args_type = [arg.type for arg in func.body.block.args]
    new_function_type = FunctionType.from_lists(
        new_args_type, func.function_type.outputs
    )
    func.function_type = new_function_type

    lowerToSMTModule(module, width, ctx)
    resultFunc = module.ops.first
    assert isinstance(resultFunc, DefineFunOp)
    smt_func_type = resultFunc.func_type
    ret_val_type = resultFunc.return_val.type
    new_smt_func_type = FunctionType.from_lists(smt_func_type.inputs, [ret_val_type])
    resultFunc.ret.type = new_smt_func_type
    return resultFunc


def verify_smt_transfer_function(
    smt_transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    width: int,
    transfer_function: TransferFunction,
    func_name_to_func: dict[str, FuncOp],
    dynamic_transfer_functions: dict[str, Func],
) -> None:
    # Soundness check
    query_module = ModuleOp([])
    dynamic_concrete_function_module = ModuleOp([])
    dynamic_transfer_function_module = ModuleOp([])
    int_attr = generateIntAttrArg(smt_transfer_function.int_attr_arg)

    # enumerating all possible int attr
    while True:
        query_module = ModuleOp([])
        added_ops = int_attr_check(
            smt_transfer_function, domain_constraint, instance_constraint, int_attr
        )
        query_module.body.block.add_ops(added_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)
        LowerToSMT().apply(ctx, query_module)
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
                )

            assert smt_transfer_function.transfer_function is not None

            query_module = ModuleOp([])
            if smt_transfer_function.is_forward:
                added_ops = forward_soundness_check(
                    smt_transfer_function,
                    domain_constraint,
                    instance_constraint,
                    int_attr,
                )
            else:
                added_ops = backward_soundness_check(
                    smt_transfer_function,
                    domain_constraint,
                    instance_constraint,
                    int_attr,
                )
            query_module.body.block.add_ops(added_ops)
            FunctionCallInline(True, {}).apply(ctx, query_module)
            LowerToSMT().apply(ctx, query_module)
            # print_to_smtlib(query_module, sys.stdout)

            print("Soundness Check result:", verify_pattern(ctx, query_module))

            if smt_transfer_function.soundness_counterexample is not None:
                query_module = ModuleOp([])
                counterexample_func_name = (
                    smt_transfer_function.soundness_counterexample.fun_name.data
                )
                added_ops = counterexample_check(
                    func_name_to_func[counterexample_func_name],
                    smt_transfer_function.soundness_counterexample,
                    domain_constraint,
                    int_attr,
                )
                query_module.body.block.add_ops(added_ops)
                FunctionCallInline(True, {}).apply(ctx, query_module)
                LowerToSMT().apply(ctx, query_module)

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
    domain_constraint: FunctionCollection = None
    instance_constraint: FunctionCollection = None
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
                domain_constraint = FunctionCollection(func, create_smt_function)
            elif func_name == INSTANCE_CONSTRAINT:
                assert instance_constraint is None
                instance_constraint = FunctionCollection(func, create_smt_function)

    FunctionCallInline(False, func_name_to_func).apply(ctx, module)

    for width in solveVectorWidth():
        print("Current width: ", width)
        smt_module = module.clone()

        # expand for loops
        unrollTransferLoop = UnrollTransferLoop(width)
        assert isinstance(smt_module, ModuleOp)
        unrollTransferLoop.apply(ctx, smt_module)
        concrete_funcs: list[FuncOp] = []
        transfer_function_name_to_concrete_function_name: dict[str, str] = {}
        dynamic_transfer_functions: dict[str, FuncOp] = {}

        # add concrete operations for every transfer functions
        for op in smt_module.ops:
            # op is a transfer function
            if isinstance(op, FuncOp) and "applied_to" in op.attributes:
                concrete_func_name = op.attributes["applied_to"].data[0].data
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

                if concrete_func_name not in concrete_funcs:
                    extra = None
                    if len(op.attributes["applied_to"].data) > 1:
                        extra = op.attributes["applied_to"].data[1]
                        assert (
                            isinstance(extra, IntegerAttr)
                            and "only support for integer attr for the second applied arg for now"
                        )
                        extra = extra.value.data
                    concrete_funcs.append(
                        get_concrete_function(concrete_func_name, width, extra)
                    )
                    transfer_function_name_to_concrete_function_name[
                        func_name
                    ] = concrete_funcs[-1].sym_name.data

        smt_module.body.block.add_ops(concrete_funcs)
        lowerToSMTModule(smt_module, width, ctx)

        func_name_to_smt_func: dict[str, DefineFunOp] = {}
        for op in smt_module.ops:
            if isinstance(op, DefineFunOp):
                func_name = op.fun_name.data
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
                abs_op_constraint = func_name_to_smt_func[
                    transfer_function.transfer_function.attributes[
                        "abs_op_constraint"
                    ].data
                ]

            op_constraint = None
            if "op_constraint" in transfer_function.transfer_function.attributes:
                op_constraint = func_name_to_smt_func[
                    transfer_function.transfer_function.attributes["op_constraint"].data
                ]

            soundness_counterexample = None
            if (
                "soundness_counterexample"
                in transfer_function.transfer_function.attributes
            ):
                soundness_counterexample = func_name_to_smt_func[
                    transfer_function.transfer_function.attributes[
                        "soundness_counterexample"
                    ].data
                ]

            int_attr_arg = None
            int_attr_constraint = None
            if "int_attr" in transfer_function.transfer_function.attributes:
                assert (
                    "int_attr_constraint"
                    in transfer_function.transfer_function.attributes
                )
                int_attr_arg = get_int_attr_arg(transfer_function.transfer_function)
                int_attr_constraint = func_name_to_smt_func[
                    transfer_function.transfer_function.attributes[
                        "int_attr_constraint"
                    ].data
                ]

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
            )
