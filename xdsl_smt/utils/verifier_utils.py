import subprocess

from xdsl.context import MLContext

from io import StringIO

from xdsl.utils.hints import isa
from ..dialects.smt_dialect import (
    DefineFunOp,
)
from xdsl_smt.dialects.transfer import (
    AbstractValueType,
    TransIntegerType,
    TupleType,
)
from xdsl.ir.core import Attribute
from xdsl.dialects.builtin import (
    ModuleOp,
    IntegerAttr,
    IntegerType,
    i1,
    FunctionType,
    ArrayAttr,
    StringAttr,
)
from xdsl.dialects.func import FuncOp, ReturnOp
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
)

from ..utils.transfer_function_check_util import (
    forward_soundness_check,
    backward_soundness_check,
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


def solve_vector_width(maximal_bits: int):
    return list(range(1, maximal_bits))


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


def get_concrete_function(
    concrete_op_name: str, width: int, extra: int | None
) -> FuncOp:
    """
    Given a name of one concrete operation, return a function with only that operation
    """

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
                func_name += str(extra)
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
                funcTy = FunctionType.from_lists([intTy, intTy], [intTy])
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


def lower_to_smt_module(module: ModuleOp, width: int, ctx: MLContext):
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
    """
    Input: a function with type FuncOp
    Return: True if the function is a transfer function that needs to be verified
            False if the function is a helper function or others
    """

    return "applied_to" in func.attributes


def is_forward(func: FuncOp) -> bool:
    """
    Input: a transfer function with type FuncOp
    Return: True if the transfer function is a forward transfer function
            False if the transfer function is a backward transfer function
    """

    if "is_forward" in func.attributes:
        forward = func.attributes["is_forward"]
        assert isinstance(forward, IntegerAttr)
        return forward.value.data != 0
    return False


def generate_int_attr_arg(int_attr_arg: list[int] | None) -> dict[int, int]:
    """
    Input: a list describes locations of args of integer attributes
    Return: a dictionary with init all integer attributes to zeros

    Example: [1,2] -> {1: 0, 2: 0} ; [] -> {}
    """

    if int_attr_arg is None:
        return {}
    intAttr: dict[int, int] = {}
    for i in int_attr_arg:
        intAttr[i] = 0
    return intAttr


INSTANCE_CONSTRAINT = "getInstanceConstraint"
DOMAIN_CONSTRAINT = "getConstraint"
TMP_MODULE: list[ModuleOp] = []


def create_smt_function(func: FuncOp, width: int, ctx: MLContext) -> DefineFunOp:
    """
    Input: a function with type FuncOp
    Return: the function lowered to SMT dialect with specified width

    We might reuse some function with specific width so we save it to global TMP_MODULE
    Class FunctionCollection is the only caller of this function and maintains all generated SMT functions
    """

    global TMP_MODULE
    TMP_MODULE.append(ModuleOp([func.clone()]))
    lower_to_smt_module(TMP_MODULE[-1], width, ctx)
    resultFunc = TMP_MODULE[-1].ops.first
    assert isinstance(resultFunc, DefineFunOp)
    return resultFunc


def soundness_check(
    smt_transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    int_attr: dict[int, int],
    ctx: MLContext,
) -> bool:
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

    result = verify_pattern(ctx, query_module)
    return result


def verify_smt_transfer_function(
    smt_transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    ctx: MLContext,
) -> bool:
    # Soundness check
    int_attr = generate_int_attr_arg(smt_transfer_function.int_attr_arg)
    # assert current use has no int_attr
    assert int_attr == {}

    assert smt_transfer_function.concrete_function is not None
    assert smt_transfer_function.transfer_function is not None

    soundness_result = soundness_check(
        smt_transfer_function,
        domain_constraint,
        instance_constraint,
        int_attr,
        ctx,
    )
    if not soundness_result:
        return False
    return True


def build_init_module(
    transfer_function: FuncOp, helper_funcs: list[FuncOp], ctx: MLContext
):
    func_name_to_func: dict[str, FuncOp] = {}
    module_op = ModuleOp([])
    module_op.body.block.add_ops(
        [transfer_function.clone()] + [func.clone() for func in helper_funcs]
    )
    domain_constraint: FunctionCollection | None = None
    instance_constraint: FunctionCollection | None = None
    transfer_function_obj: TransferFunction | None = None
    transfer_function_name = transfer_function.sym_name.data
    for func in module_op.ops:
        assert isinstance(func, FuncOp)
        func_name = func.sym_name.data
        func_name_to_func[func_name] = func

        # Check func validity
        assert len(func.function_type.inputs) == len(func.args)
        for func_type_arg, arg in zip(func.function_type.inputs, func.args):
            assert func_type_arg == arg.type
        return_op = func.body.block.last_op
        assert return_op is not None and isinstance(return_op, ReturnOp)
        assert return_op.operands[0].type == func.function_type.outputs.data[0]
        # End of check function type

        if func_name == transfer_function_name:
            assert transfer_function_obj is None
            transfer_function_obj = TransferFunction(
                func,
            )
        if func_name == DOMAIN_CONSTRAINT:
            assert domain_constraint is None
            domain_constraint = FunctionCollection(func, create_smt_function, ctx)
        elif func_name == INSTANCE_CONSTRAINT:
            assert instance_constraint is None
            instance_constraint = FunctionCollection(func, create_smt_function, ctx)

    assert domain_constraint is not None
    assert instance_constraint is not None
    assert transfer_function_obj is not None

    func_name_to_func[transfer_function.sym_name.data] = transfer_function
    if len(func_name_to_func) != len(helper_funcs) + 1:
        print(
            [func.sym_name.data for func in helper_funcs]
            + [transfer_function.sym_name.data]
        )
        raise ValueError("Found function with the same name in the input")
    return (
        module_op,
        func_name_to_func,
        transfer_function_obj,
        domain_constraint,
        instance_constraint,
    )


def verify_transfer_function(
    transfer_function: FuncOp,
    helper_funcs: list[FuncOp],
    ctx: MLContext,
    maximal_verify_bits: int = 32,
) -> bool:
    (
        module_op,
        func_name_to_func,
        transfer_function_obj,
        domain_constraint,
        instance_constraint,
    ) = build_init_module(transfer_function, helper_funcs, ctx)

    FunctionCallInline(False, func_name_to_func).apply(ctx, module_op)

    for width in solve_vector_width(maximal_verify_bits):
        smt_module = module_op.clone()

        # expand for loops
        unrollTransferLoop = UnrollTransferLoop(width)
        assert isinstance(smt_module, ModuleOp)
        unrollTransferLoop.apply(ctx, smt_module)
        concrete_func_name: str | None = None
        concrete_func: FuncOp | None = None
        for op in smt_module.ops:
            # op is a transfer function
            if isinstance(op, FuncOp) and "applied_to" in op.attributes:
                assert isa(
                    applied_to := op.attributes["applied_to"], ArrayAttr[Attribute]
                )
                assert isinstance(applied_to.data[0], StringAttr)
                concrete_func_name = applied_to.data[0].data

                extra = None
                assert isa(
                    applied_to := op.attributes["applied_to"], ArrayAttr[Attribute]
                )
                assert isinstance(applied_to.data[0], StringAttr)
                if len(applied_to.data) > 1:
                    extra = applied_to.data[1]
                    assert (
                        isinstance(extra, IntegerAttr)
                        and "only support for integer attr for the second applied arg for now"
                    )
                    extra = extra.value.data
                concrete_func = get_concrete_function(concrete_func_name, width, extra)
                concrete_func_name = concrete_func.sym_name.data

                if len(applied_to.data) >= 2:
                    concrete_func_name += str(extra)

        assert concrete_func is not None
        assert concrete_func_name is not None
        smt_module.body.block.add_op(concrete_func)
        lower_to_smt_module(smt_module, width, ctx)

        func_name_to_smt_func: dict[str, DefineFunOp] = {}
        for op in smt_module.ops:
            if isinstance(op, DefineFunOp):
                op_func_name = op.fun_name
                assert op_func_name is not None
                func_name = op_func_name.data
                func_name_to_smt_func[func_name] = op

        func_name = transfer_function.sym_name.data
        assert func_name is not None

        smt_concrete_func = None
        if concrete_func_name in func_name_to_smt_func:
            smt_concrete_func = func_name_to_smt_func[concrete_func_name]
        assert smt_concrete_func is not None

        smt_transfer_function = None
        if func_name in func_name_to_smt_func:
            smt_transfer_function = func_name_to_smt_func[func_name]

        abs_op_constraint = None
        if "abs_op_constraint" in transfer_function_obj.transfer_function.attributes:
            abs_op_constraint_name_attr = (
                transfer_function_obj.transfer_function.attributes["abs_op_constraint"]
            )
            assert isinstance(abs_op_constraint_name_attr, StringAttr)
            abs_op_constraint_name = abs_op_constraint_name_attr.data
            abs_op_constraint = func_name_to_smt_func[abs_op_constraint_name]

        op_constraint = None
        if "op_constraint" in transfer_function_obj.transfer_function.attributes:
            op_constraint_func_name_attr = (
                transfer_function_obj.transfer_function.attributes["op_constraint"]
            )
            assert isinstance(op_constraint_func_name_attr, StringAttr)
            op_constraint_func_name = op_constraint_func_name_attr.data
            op_constraint = func_name_to_smt_func[op_constraint_func_name]

        soundness_counterexample = None
        int_attr_arg = None
        int_attr_constraint = None

        smt_transfer_function_obj = SMTTransferFunction(
            transfer_function_obj,
            func_name,
            concrete_func_name,
            abs_op_constraint,
            op_constraint,
            soundness_counterexample,
            int_attr_arg,
            int_attr_constraint,
            smt_transfer_function,
            smt_concrete_func,
        )

        result = verify_smt_transfer_function(
            smt_transfer_function_obj,
            domain_constraint,
            instance_constraint,
            ctx,
        )
        if not result:
            return False
    return True
