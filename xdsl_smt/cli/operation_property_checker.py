from xdsl_smt.dialects.smt_dialect import ConstantBoolOp, YieldOp, ForallOp
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
from ..dialects.smt_utils_dialect import SMTUtilsDialect, FirstOp
from xdsl.ir import Block, Region, SSAValue
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
from ..utils.transfer_function_util import (
    SMTTransferFunction,
    FunctionCollection,
    TransferFunction,
    fixDefiningOpReturnType,
    getArgumentWidthsWithEffect,
    getArgumentInstancesWithEffect,
    callFunctionWithEffect,
    insertArgumentInstancesToBlockWithEffect,
)

from ..utils.transfer_function_check_util import (
    forward_soundness_check,
    backward_soundness_check,
    counterexample_check,
    int_attr_check,
    forward_precision_check,
    module_op_validity_check,
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


def check_commutativity(func: DefineFunOp, ctx: MLContext) -> bool:
    if len(func.body.block.args) == 3:
        query_module = ModuleOp([])
        # getInstance
        effect = ConstantBoolOp(False)
        arg_ops = getArgumentInstancesWithEffect(func, {})
        args = [arg.res for arg in arg_ops]

        # func(x, y)
        call_func_x_y_op, call_func_x_y_first_op = callFunctionWithEffect(
            func, args, effect.res
        )

        # func(y, x)
        call_func_y_x_op, call_func_y_x_first_op = callFunctionWithEffect(
            func, [args[1], args[0]], effect.res
        )

        # assert func(y, x) != func(x,y)
        call_distinct_op = DistinctOp(
            call_func_x_y_first_op.res, call_func_y_x_first_op.res
        )
        assert_op = AssertOp(call_distinct_op.res)

        all_ops = (
            arg_ops
            + [call_func_x_y_op, call_func_x_y_first_op]
            + [call_func_y_x_op, call_func_y_x_first_op]
            + [call_distinct_op, assert_op, CheckSatOp()]
        )

        query_module.body.block.add_ops(all_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)

        result = verify_pattern(ctx, query_module)
        print("Commutativity Check result:", result)
        return result
    return False


def check_associativity(func: DefineFunOp, ctx: MLContext) -> bool:
    if len(func.body.block.args) == 3:
        query_module = ModuleOp([])
        # getInstance x, y, z, useless
        effect = ConstantBoolOp(False)
        arg_ops = getArgumentInstancesWithEffect(func, {})
        arg_ops_again = getArgumentInstancesWithEffect(func, {})
        args = [arg.res for arg in arg_ops] + [arg.res for arg in arg_ops_again]

        # func(x, y)
        call_func_x_y_op, call_func_x_y_first_op = callFunctionWithEffect(
            func, [args[0], args[1]], effect.res
        )
        # func(func(x, y), z)
        call_func_func_x_y_z_op, call_func_func_x_y_z_first_op = callFunctionWithEffect(
            func, [call_func_x_y_first_op.res, args[2]], effect.res
        )

        # func(y, z)
        call_func_y_z_op, call_func_y_z_first_op = callFunctionWithEffect(
            func, [args[1], args[2]], effect.res
        )
        # func(x, func(y, z)
        call_func_x_func_y_z_op, call_func_x_func_y_z_first_op = callFunctionWithEffect(
            func, [args[0], call_func_y_z_first_op.res], effect.res
        )

        # assert func(func(x, y), z) == func(x,func(y, z))
        call_distinct_op = DistinctOp(
            call_func_func_x_y_z_first_op.res, call_func_x_func_y_z_first_op.res
        )
        assert_op = AssertOp(call_distinct_op.res)

        all_ops = (
            arg_ops
            + arg_ops_again
            + [call_func_x_y_op, call_func_x_y_first_op]
            + [call_func_func_x_y_z_op, call_func_func_x_y_z_first_op]
            + [call_func_y_z_op, call_func_y_z_first_op]
            + [call_func_x_func_y_z_op, call_func_x_func_y_z_first_op]
            + [call_distinct_op, assert_op, CheckSatOp()]
        )

        query_module.body.block.add_ops(all_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)

        result = verify_pattern(ctx, query_module)
        print("Associativity Check result:", result)
        return result
    return False


def check_involution(func: DefineFunOp, ctx: MLContext) -> bool:
    if len(func.body.block.args) == 2:
        query_module = ModuleOp([])
        # getInstance x
        effect = ConstantBoolOp(False)
        arg_ops = getArgumentInstancesWithEffect(func, {})
        args = [arg.res for arg in arg_ops]
        # func(func(x)) == x

        # func(x)
        call_func_x_op, call_func_x_first_op = callFunctionWithEffect(
            func, args, effect.res
        )
        # func(func(x))
        call_func_func_x_op, call_func_func_x_first_op = callFunctionWithEffect(
            func, [call_func_x_first_op.res], effect.res
        )

        # assert func(func(x)) != x
        call_distinct_op = DistinctOp(call_func_func_x_first_op.res, args[0])
        assert_op = AssertOp(call_distinct_op.res)

        all_ops = (
            arg_ops
            + [call_func_x_op, call_func_x_first_op]
            + [call_func_func_x_op, call_func_func_x_first_op]
            + [call_distinct_op, assert_op, CheckSatOp()]
        )

        query_module.body.block.add_ops(all_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)

        result = verify_pattern(ctx, query_module)
        print("Involution Check result:", result)
        return result
    return False


def check_idempotence(func: DefineFunOp, ctx: MLContext) -> bool:
    # func(x, x) == x
    if len(func.body.block.args) == 3:
        query_module = ModuleOp([])
        # getInstance x, y
        effect = ConstantBoolOp(False)
        arg_ops = getArgumentInstancesWithEffect(func, {})
        args = [arg.res for arg in arg_ops]

        # assert x == y
        x_eq_y_op = EqOp(args[0], args[1])
        assert_x_eq_y_op = AssertOp(x_eq_y_op.res)

        # func(x, x)
        call_func_x_x_op, call_func_x_x_first_op = callFunctionWithEffect(
            func, args, effect.res
        )

        # assert func(y, x) != x
        call_distinct_op = DistinctOp(call_func_x_x_first_op.res, args[0])
        assert_op = AssertOp(call_distinct_op.res)

        all_ops = (
            arg_ops
            + [x_eq_y_op, assert_x_eq_y_op]
            + [call_func_x_x_op, call_func_x_x_first_op]
            + [call_distinct_op, assert_op, CheckSatOp()]
        )

        query_module.body.block.add_ops(all_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)

        result = verify_pattern(ctx, query_module)
        print("Idempotence Check result:", result)
        return result
    return False


# Forall([x], op(x,ele)==ele)
def get_forall_absorbing_property(
    op: DefineFunOp, ele: SSAValue, is_first_operand: bool, effect: ConstantBoolOp
) -> ForallOp:
    forall_block = Block()
    forall_block_args = insertArgumentInstancesToBlockWithEffect(op, {}, forall_block)

    if is_first_operand:
        op_args = [ele, forall_block_args[0]]
    else:
        op_args = [forall_block_args[0], ele]

    # func(x, ele)
    call_func_x_ele_op, call_func_x_ele_first_op = callFunctionWithEffect(
        op, op_args, effect.res
    )

    call_func_x_ele_first_first_op = FirstOp(call_func_x_ele_first_op.res)
    ele_first_op = FirstOp(ele)

    # func(x,ele)==ele
    call_func_x_ele_eq_ele_op = EqOp(
        call_func_x_ele_first_first_op.res, ele_first_op.res
    )

    yield_op = YieldOp(call_func_x_ele_eq_ele_op.res)

    forall_all_ops = (
        [call_func_x_ele_op, call_func_x_ele_first_op]
        + [call_func_x_ele_first_first_op, ele_first_op]
        + [
            call_func_x_ele_eq_ele_op,
            yield_op,
        ]
    )
    forall_block.add_ops(forall_all_ops)

    forall_op = ForallOp.from_variables([], Region(forall_block))
    return forall_op


# Forall([x], op(x,ele)==ele)
# Forall([x], op(ele,x)==ele)
def check_absorbing_element(func: DefineFunOp, ctx: MLContext, commutativity: bool):
    if len(func.body.block.args) == 3:
        operand_order = [True]
        if not commutativity:
            operand_order.append(False)

        final_result = False
        for op_order in operand_order:
            query_module = ModuleOp([])
            # getInstance x, y
            effect = ConstantBoolOp(False)
            arg_ops = getArgumentInstancesWithEffect(func, {})
            args = [arg.res for arg in arg_ops]

            forall_op = get_forall_absorbing_property(func, args[0], op_order, effect)

            # assert forall
            assert_forall_op = AssertOp(forall_op.res)

            all_ops = arg_ops + [forall_op, assert_forall_op] + [CheckSatOp()]

            query_module.body.block.add_ops(all_ops)
            FunctionCallInline(True, {}).apply(ctx, query_module)

            result, result_str = get_model(ctx, query_module)

            def print_absorbing_element(
                result: bool, func_name: str, s: str, is_first_operand: bool
            ) -> str:
                if result:
                    lines = s.split("\n")
                    for line in lines:
                        if line.startswith("#x"):
                            int_val = int(line.replace("#", "0"), 16)
                            return "{func_name}({first_op}, {second_op}) == {int_val}".format(
                                func_name=func_name,
                                first_op=(int_val if is_first_operand else "x"),
                                second_op=(int_val if not is_first_operand else "x"),
                                int_val=int_val,
                            )

                return "N/A"

            print(
                "Absorbing Element Check result: ",
                print_absorbing_element(
                    result, func.fun_name.data, result_str, op_order
                ),
            )
            final_result |= result
        return final_result
    return False


# Forall([x], op(x,ele)==x)
def get_forall_identity_property(
    op: DefineFunOp, ele: SSAValue, is_first_operand: bool, effect: ConstantBoolOp
) -> ForallOp:
    forall_block = Block()
    forall_block_args = insertArgumentInstancesToBlockWithEffect(op, {}, forall_block)

    if is_first_operand:
        op_args = [ele, forall_block_args[0]]
    else:
        op_args = [forall_block_args[0], ele]

    # func(x, ele)
    call_func_x_ele_op, call_func_x_ele_first_op = callFunctionWithEffect(
        op, op_args, effect.res
    )

    # remove poison
    call_func_x_ele_first_first_op = FirstOp(call_func_x_ele_first_op.res)
    x_first_op = FirstOp(forall_block_args[0])

    # func(x,ele)==x
    call_func_x_ele_eq_x_op = EqOp(call_func_x_ele_first_first_op.res, x_first_op.res)

    yield_op = YieldOp(call_func_x_ele_eq_x_op.res)

    forall_all_ops = (
        [call_func_x_ele_op, call_func_x_ele_first_op]
        + [call_func_x_ele_first_first_op, x_first_op]
        + [
            call_func_x_ele_eq_x_op,
            yield_op,
        ]
    )
    forall_block.add_ops(forall_all_ops)

    forall_op = ForallOp.from_variables([], Region(forall_block))
    return forall_op


# Forall([x], op(x,ele)==x)
# Forall([x], op(ele,x)==x)
def check_identity_element(func: DefineFunOp, ctx: MLContext, commutativity: bool):
    if len(func.body.block.args) == 3:
        operand_order = [True]
        if not commutativity:
            operand_order.append(False)

        final_result = False
        for op_order in operand_order:
            query_module = ModuleOp([])
            # getInstance x, y
            effect = ConstantBoolOp(False)
            arg_ops = getArgumentInstancesWithEffect(func, {})
            args = [arg.res for arg in arg_ops]

            forall_op = get_forall_identity_property(func, args[0], op_order, effect)

            # assert forall
            assert_forall_op = AssertOp(forall_op.res)

            all_ops = arg_ops + [forall_op, assert_forall_op] + [CheckSatOp()]

            query_module.body.block.add_ops(all_ops)
            FunctionCallInline(True, {}).apply(ctx, query_module)

            result, result_str = get_model(ctx, query_module)

            def print_zero_element(
                result: bool, func_name: str, s: str, is_first_operand: bool
            ) -> str:
                if result:
                    lines = s.split("\n")
                    for line in lines:
                        if line.startswith("#x"):
                            int_val = int(line.replace("#", "0"), 16)
                            return "{func_name}({first_op}, {second_op}) == x".format(
                                func_name=func_name,
                                first_op=(int_val if is_first_operand else "x"),
                                second_op=(int_val if not is_first_operand else "x"),
                            )

                return "N/A"

            print(
                "Identity Element Check result: ",
                print_zero_element(result, func.fun_name.data, result_str, op_order),
            )
            final_result |= result
        return final_result
    return False


# Forall([x], op(x,x)==ele)
def get_forall_self_annihilation_property(
    op: DefineFunOp, ele: SSAValue, effect: ConstantBoolOp
) -> ForallOp:
    forall_block = Block()
    forall_block_args = insertArgumentInstancesToBlockWithEffect(op, {}, forall_block)

    # func(x, x)
    call_func_x_x_op, call_func_x_x_first_op = callFunctionWithEffect(
        op, [forall_block_args[0], forall_block_args[0]], effect.res
    )

    call_func_x_x_first_first_op = FirstOp(call_func_x_x_first_op.res)
    ele_first_op = FirstOp(ele)

    # func(x,x)==ele
    call_func_x_x_eq_ele_op = EqOp(call_func_x_x_first_first_op.res, ele_first_op.res)

    yield_op = YieldOp(call_func_x_x_eq_ele_op.res)

    forall_all_ops = (
        [call_func_x_x_op, call_func_x_x_first_op]
        + [call_func_x_x_first_first_op, ele_first_op]
        + [
            call_func_x_x_eq_ele_op,
            yield_op,
        ]
    )
    forall_block.add_ops(forall_all_ops)

    forall_op = ForallOp.from_variables([], Region(forall_block))
    return forall_op


def check_self_annihilation(func: DefineFunOp, ctx: MLContext):
    if len(func.body.block.args) == 3:
        query_module = ModuleOp([])
        # getInstance x, y
        effect = ConstantBoolOp(False)
        arg_ops = getArgumentInstancesWithEffect(func, {})
        args = [arg.res for arg in arg_ops]

        forall_op = get_forall_self_annihilation_property(func, args[0], effect)

        # assert forall
        assert_forall_op = AssertOp(forall_op.res)

        all_ops = arg_ops + [forall_op, assert_forall_op] + [CheckSatOp()]

        query_module.body.block.add_ops(all_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)

        result, result_str = get_model(ctx, query_module)

        def print_self_annihilation_element(
            result: bool, func_name: str, s: str
        ) -> str:
            if result:
                lines = s.split("\n")
                for line in lines:
                    if line.startswith("#x"):
                        int_val = int(line.replace("#", "0"), 16)
                        return "{func_name}(x, x) == {int_val}".format(
                            func_name=func_name, int_val=int_val
                        )
            return "N/A"

        print(
            "Self Annihilation Element Check result: ",
            print_self_annihilation_element(result, func.fun_name.data, result_str),
        )
        return result
    return False


def check_all_property(func: DefineFunOp, ctx: MLContext):
    comm = check_commutativity(func, ctx)
    check_associativity(func, ctx)
    check_involution(func, ctx)
    check_idempotence(func, ctx)
    check_absorbing_element(func, ctx, comm)
    check_identity_element(func, ctx, comm)
    check_self_annihilation(func, ctx)

    pass


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
    assert isinstance(module, ModuleOp)

    smt_module = module.clone()
    assert isinstance(smt_module, ModuleOp)
    lowerToSMTModule(smt_module, 8, ctx)
    for op in smt_module.ops:
        if isinstance(op, DefineFunOp):
            print("Current check: ", op.fun_name)
            check_all_property(op, ctx)
            print("========")
