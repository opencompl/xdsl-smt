"""
This file contains semantics of constraints and rewrites for integer arithmetic
in PDL.
"""

from typing import Callable
from xdsl.ir import ErasedSSAValue, Operation, SSAValue
from xdsl.utils.hints import isa
from xdsl.pattern_rewriter import PatternRewriter

from xdsl.dialects.pdl import ApplyNativeConstraintOp, ApplyNativeRewriteOp
import xdsl_smt.dialects.smt_bitvector_dialect as smt_bv
import xdsl_smt.dialects.smt_utils_dialect as smt_utils
import xdsl_smt.dialects.smt_dialect as smt
from xdsl_smt.passes.pdl_to_smt import PDLToSMTRewriteContext


def single_op_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, op_type: type[Operation]
) -> None:
    lhs, rhs = op.args
    new_op = op_type.create(operands=[lhs, rhs], result_types=[lhs.type])
    rewriter.replace_matched_op(new_op)


def addi_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    return single_op_rewrite(op, rewriter, smt_bv.AddOp)


def subi_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    return single_op_rewrite(op, rewriter, smt_bv.SubOp)


def muli_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    return single_op_rewrite(op, rewriter, smt_bv.MulOp)


def get_zero_attr_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    (value,) = op.args
    assert isinstance(value, ErasedSSAValue)
    type = context.pdl_types_to_types[value.old_value]

    width: int
    # Poison case
    if isa(type, smt_utils.PairType[smt_bv.BitVectorType, smt.BoolType]):
        width = type.first.width.data
    elif isinstance(type, smt_bv.BitVectorType):
        width = type.width.data
    else:
        raise Exception(
            "get_zero_attr expects the input to be lowered to a `!smt.bv<...>` or a"
            "!smt.utils.pair<!smt.bv<...>, !smt.bool>."
        )

    zero = smt_bv.ConstantOp(0, width)
    rewriter.replace_matched_op([zero])


def is_minus_one(
    op: ApplyNativeConstraintOp,
    rewriter: PatternRewriter,
    context: PDLToSMTRewriteContext,
) -> SSAValue:
    (value,) = op.args

    if not isinstance(value.type, smt_bv.BitVectorType):
        raise Exception(
            "is_minus_one expects the input to be lowered to a `!smt.bv<...>`"
        )

    width = value.type.width.data
    minus_one = smt_bv.ConstantOp(2**width - 1, width)
    eq_minus_one = smt.EqOp(value, minus_one.res)
    rewriter.replace_matched_op([eq_minus_one, minus_one], [])
    return eq_minus_one.res


integer_arith_native_rewrites: dict[
    str,
    Callable[[ApplyNativeRewriteOp, PatternRewriter, PDLToSMTRewriteContext], None],
] = {
    "addi": addi_rewrite,
    "subi": subi_rewrite,
    "muli": muli_rewrite,
    "get_zero_attr": get_zero_attr_rewrite,
}

integer_arith_native_constraints = {
    "is_minus_one": is_minus_one,
}
