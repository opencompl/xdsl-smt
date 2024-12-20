from xdsl_smt.dialects.smt_dialect import ConstantBoolOp, YieldOp, ForallOp, EvalOp
from xdsl.dialects.func import FuncOp
import argparse
import subprocess

from xdsl.context import MLContext
from xdsl.parser import Parser

from io import StringIO

from xdsl.utils.hints import isa
from ..dialects.smt_dialect import (
    SMTDialect,
    DefineFunOp,
    EqOp,
    AssertOp,
    CheckSatOp,
    DistinctOp,
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
from ..dialects.smt_utils_dialect import SMTUtilsDialect, FirstOp, AnyPairType, SecondOp
from xdsl.ir import Block, Region, SSAValue, BlockArgument, Attribute
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
from ..utils.transfer_function_util import (
    getArgumentWidthsWithEffect,
    getResultWidth,
    FunctionCollection,
    SMTTransferFunction,
    getArgumentInstancesWithEffect,
    callFunctionAndAssertResultWithEffect,
    TransferFunction,
    fixDefiningOpReturnType,
    callFunctionWithEffect,
)


def verify_pattern(ctx: MLContext, op: ModuleOp) -> bool:
    # cloned_op = op.clone()
    cloned_op = op
    stream = StringIO()
    LowerPairs().apply(ctx, cloned_op)
    CanonicalizePass().apply(ctx, cloned_op)
    DeadCodeElimination().apply(ctx, cloned_op)

    print_to_smtlib(cloned_op, stream)
    print(stream.getvalue())
    res = subprocess.run(
        ["z3", "-in"],
        capture_output=True,
        input=stream.getvalue(),
        text=True,
    )
    if res.returncode != 0:
        raise Exception(res.stderr)
    print(res.stdout)
    return "unsat" in res.stdout


def get_model(ctx: MLContext, op: ModuleOp) -> str:
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
        assert False
        return ""
    return str(res.stdout)


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


SYNTH_WIDTH = 8
TEST_SET_SIZE = 10
CONCRETE_VAL_PER_TEST_CASE = 2
INSTANCE_CONSTRAINT = "getInstanceConstraint"
DOMAIN_CONSTRAINT = "getConstraint"
TMP_MODULE: list[ModuleOp] = []
ctx: MLContext


def getTestSetQueryModel(
    transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    int_attr: dict[int, int],
) -> ModuleOp:
    abstract_func = transfer_function.transfer_function
    concrete_func = transfer_function.concrete_function
    abs_op_constraint = transfer_function.abstract_constraint
    op_constraint = transfer_function.op_constraint
    is_abstract_arg = transfer_function.is_abstract_arg

    query_model = ModuleOp([])
    query_module_block = query_model.body.block
    effect = ConstantBoolOp(False)
    constant_bv_1 = ConstantOp(1, 1)

    arg_widths = getArgumentWidthsWithEffect(concrete_func)
    result_width = getResultWidth(concrete_func)

    # We need TEST_SET_SIZE of abstract values
    # For each abstract values, we need concrete_val_per_test_case
    abs_arg_ops_collection: list[list[Operation]] = []
    abs_arg_ssa_values_collection: list[list[SSAValue]] = []
    abs_domain_constraints_ops_collection: list[list[Operation]] = []

    abs_arg_include_crt_arg_constraints_ops_collection: list[list[Operation]] = []
    abs_args_distinct_ops_constraints_collection: list[list[Operation]] = []

    crt_arg_ops_collection: list[list[Operation]] = []
    crt_args_first_ops_collection: list[list[Operation]] = []
    crt_args_ssa_values_collection: list[list[list[SSAValue]]] = []

    crt_res_ops_collection: list[list[Operation]] = []
    crt_res_ssa_values_collection: list[list[SSAValue]] = []

    # Each iteration generates one test case
    for i in range(TEST_SET_SIZE):
        abs_arg_ops = getArgumentInstancesWithEffect(abstract_func, int_attr)
        abs_args: list[SSAValue] = [arg.res for arg in abs_arg_ops]

        query_module_block.add_ops(abs_arg_ops)
        abs_arg_ssa_values_collection.append(abs_args)
        crt_args_ssa_values_collection.append([])
        crt_res_ssa_values_collection.append([])

        for i, abs_arg in enumerate(abs_args):
            if is_abstract_arg[i]:
                query_module_block.add_ops(
                    callFunctionAndAssertResultWithEffect(
                        domain_constraint.getFunctionByWidth(arg_widths[i]),
                        [abs_arg],
                        constant_bv_1,
                        effect.res,
                    )
                )
                """
                abs_domain_constraints_ops_collection.append(callFunctionAndAssertResultWithEffect(
                    domain_constraint.getFunctionByWidth(arg_widths[i]),
                    [abs_arg],
                    constant_bv_1,
                    effect.res,
                ))
                """

        # Each iteration generates a set of concrete value LHS, RHS
        for crt_cnt in range(CONCRETE_VAL_PER_TEST_CASE):
            crt_arg_ops = getArgumentInstancesWithEffect(concrete_func, int_attr)
            crt_args_with_poison: list[SSAValue] = [arg.res for arg in crt_arg_ops]
            crt_arg_first_ops: list[FirstOp] = [
                FirstOp(arg) for arg in crt_args_with_poison
            ]
            crt_args: list[SSAValue] = [arg.res for arg in crt_arg_first_ops]

            query_module_block.add_ops(crt_arg_ops)
            query_module_block.add_ops(crt_arg_first_ops)
            crt_args_ssa_values_collection[-1].append(crt_args)

            for i, (abs_arg, crt_arg) in enumerate(zip(abs_args, crt_args)):
                if is_abstract_arg[i]:
                    query_module_block.add_ops(
                        callFunctionAndAssertResultWithEffect(
                            instance_constraint.getFunctionByWidth(arg_widths[i]),
                            [abs_arg, crt_arg],
                            constant_bv_1,
                            effect.res,
                        )
                    )

            call_crt_func_op, call_crt_func_first_op = callFunctionWithEffect(
                concrete_func, crt_args_with_poison, effect.res
            )
            call_crt_first_op = FirstOp(call_crt_func_first_op.res)

            query_module_block.add_ops(
                [call_crt_func_op, call_crt_func_first_op, call_crt_first_op]
            )
            crt_res_ssa_values_collection[-1].append(call_crt_first_op.res)

    abs_arg_distinct_ops: list[Operation] = []
    abs_arg_eval_depair_ops: list[Operation] = []
    abs_arg_eval_ops: list[Operation] = []

    def evalPair(val: SSAValue) -> None:
        ty: Attribute = val.type
        if isa(ty, AnyPairType):
            abs_arg_eval_depair_ops.append(FirstOp(val))
            evalPair(abs_arg_eval_depair_ops[-1].results[0])
            abs_arg_eval_depair_ops.append(SecondOp(val))
            evalPair(abs_arg_eval_depair_ops[-1].results[0])
        else:
            abs_arg_eval_ops.append(EvalOp(val))

    assert len(abs_arg_ssa_values_collection) != 0
    arity = len(abs_arg_ssa_values_collection[0])
    for ith in range(arity):
        for i in range(TEST_SET_SIZE):
            evalPair(abs_arg_ssa_values_collection[i][ith])
            for j in range(i + 1, TEST_SET_SIZE):
                distinct_op = DistinctOp(
                    abs_arg_ssa_values_collection[i][ith],
                    abs_arg_ssa_values_collection[j][ith],
                )
                assert_op = AssertOp(distinct_op.res)
                abs_arg_distinct_ops += [distinct_op, assert_op]

    crt_res_distinct_ops: list[Operation] = []
    crt_res_eval_ops: list[Operation] = []
    assert len(crt_res_ssa_values_collection) != 0
    for ith in range(TEST_SET_SIZE):
        for i in range(CONCRETE_VAL_PER_TEST_CASE):
            crt_res_eval_ops.append(EvalOp(crt_res_ssa_values_collection[ith][i]))
            for j in range(i + 1, CONCRETE_VAL_PER_TEST_CASE):
                distinct_op = DistinctOp(
                    crt_res_ssa_values_collection[ith][i],
                    crt_res_ssa_values_collection[ith][j],
                )
                assert_op = AssertOp(distinct_op.res)
                crt_res_distinct_ops += [distinct_op, assert_op]

    query_module_block.add_ops(crt_res_distinct_ops)
    query_module_block.add_ops(abs_arg_eval_depair_ops)
    query_module_block.add_ops(abs_arg_distinct_ops)

    query_module_block.add_op(CheckSatOp())
    query_module_block.add_ops(abs_arg_eval_ops)
    query_module_block.add_ops(crt_res_eval_ops)

    return query_model


def parse_smt_output_str(
    s: str, num_per_abs_val: int, arity: int
) -> list[tuple[list[list[int]], list[int]]]:
    lines = s.split("\n")
    test_set: list[tuple[list[list[int]], list[int]]] = []
    assert lines[0] == "sat"
    abs_val_list: list[list[int]] = []
    crt_val_list: list[int] = []
    tmp_list: list[int] = []
    cur_idx = 0
    for line in lines[1:]:
        cur_idx += 1
        if len(abs_val_list) == arity * TEST_SET_SIZE:
            break
        if line != "true" and line != "false":
            tmp_list.append(int(line.replace("#", "0"), 16))
            pass
        else:
            abs_val_list.append(tmp_list)
            tmp_list = []

    for line in lines[cur_idx:-1]:
        crt_val_list.append(int(line.replace("#", "0"), 16))
    assert (
        len(abs_val_list) % arity == 0 and len(abs_val_list) // arity == TEST_SET_SIZE
    )
    assert len(crt_val_list) == CONCRETE_VAL_PER_TEST_CASE * TEST_SET_SIZE
    tmp_val_list: list[list[int]] = []
    while len(crt_val_list) != 0:
        for i in range(CONCRETE_VAL_PER_TEST_CASE):
            tmp_list.append(crt_val_list.pop(0))
        for i in range(arity):
            tmp_val_list.append(abs_val_list.pop(0))
        test_set.append((tmp_val_list, tmp_list))
        tmp_list = []
        tmp_val_list = []
    return test_set


def get_transfer_function(
    module: ModuleOp, ctx: MLContext
) -> tuple[SMTTransferFunction, FunctionCollection, FunctionCollection, dict[int, int]]:
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

    for width in [SYNTH_WIDTH]:
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
            int_attr = generateIntAttrArg(smt_transfer_function_obj.int_attr_arg)
            return (
                smt_transfer_function_obj,
                domain_constraint,
                instance_constraint,
                int_attr,
            )


def generate_test_set(
    smt_transfer_function_obj: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    int_attr: dict[int, int],
    ctx: MLContext,
) -> list[tuple[list[list[int]], list[int]]]:
    query_module = getTestSetQueryModel(
        smt_transfer_function_obj, domain_constraint, instance_constraint, int_attr
    )
    FunctionCallInline(True, {}).apply(ctx, query_module)
    res = get_model(ctx, query_module)
    test_set = parse_smt_output_str(res, 2, 2)
    return test_set
