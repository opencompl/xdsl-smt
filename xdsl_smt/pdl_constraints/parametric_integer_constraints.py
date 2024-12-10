"""
This file contains semantics of constraints and rewrites for integer arithmetic
in PDL.
"""

from typing import Callable
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import Attribute, ErasedSSAValue, Operation, SSAValue
from xdsl.utils.hints import isa
from xdsl.pattern_rewriter import PatternRewriter

from xdsl.dialects.pdl import (
    ApplyNativeConstraintOp,
    ApplyNativeRewriteOp,
    TypeOp,
)
import xdsl_smt.dialects.smt_bitvector_dialect as smt_bv
import xdsl_smt.dialects.smt_int_dialect as smt_int
import xdsl_smt.dialects.smt_utils_dialect as smt_utils
import xdsl_smt.dialects.smt_dialect as smt
from typing import Callable
from xdsl_smt.passes.pdl_to_smt_context import (
    PDLToSMTRewriteContext as PDLToSMTRewriteContext,
)


def get_cst_rewrite_factory(constant: int):
    def get_cst_rewrite(
        op: ApplyNativeRewriteOp,
        rewriter: PatternRewriter,
        context: PDLToSMTRewriteContext,
    ) -> None:
        cst = smt_int.ConstantOp(constant)
        rewriter.replace_matched_op([cst])

    return get_cst_rewrite


def get_bv_type_from_optional_poison(
    type: Attribute, origin: str
) -> smt_bv.BitVectorType:
    if isa(type, smt_utils.PairType[smt_bv.BitVectorType, smt.BoolType]):
        return type.first
    elif isinstance(type, smt_bv.BitVectorType):
        return type
    else:
        raise Exception(
            f"{origin} expected to be lowered to a `!smt.bv<...>` or a "
            f"!smt.utils.pair<!smt.bv<...>, !smt.bool>. Got {type}."
        )


def single_op_rewrite(
    op: ApplyNativeRewriteOp,
    rewriter: PatternRewriter,
    op_type: type[Operation],
    fold: Callable[
        [IntegerAttr[IntegerType], IntegerAttr[IntegerType]], IntegerAttr[IntegerType]
    ],
) -> None:
    lhs, rhs = op.args
    if isinstance(lhs.owner, smt_int.ConstantOp) and isinstance(
        rhs.owner, smt_int.ConstantOp
    ):
        lhs = lhs.owner.value
        rhs = rhs.owner.value
        rewriter.replace_matched_op([smt_int.ConstantOp(fold(lhs, rhs))])
        return
    new_op = op_type.create(operands=[lhs, rhs], result_types=[lhs.type])
    rewriter.replace_matched_op(new_op)


def addi_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    def fold_addi(
        lhs: IntegerAttr[IntegerType], rhs: IntegerAttr[IntegerType]
    ) -> IntegerAttr[IntegerType]:
        raise NotImplementedError

    return single_op_rewrite(op, rewriter, smt_int.AddOp, fold_addi)


def subi_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    def fold_subi(
        lhs: IntegerAttr[IntegerType], rhs: IntegerAttr[IntegerType]
    ) -> IntegerAttr[IntegerType]:
        raise NotImplementedError

    return single_op_rewrite(op, rewriter, smt_int.SubOp, fold_subi)


def muli_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    def fold_muli(
        lhs: IntegerAttr[IntegerType], rhs: IntegerAttr[IntegerType]
    ) -> IntegerAttr[IntegerType]:
        raise NotImplementedError

    return single_op_rewrite(op, rewriter, smt_int.MulOp, fold_muli)


def integer_type_from_width(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    (width,) = op.args
    assert isinstance(width, SSAValue)
    assert isinstance(width.owner, smt_bv.ConstantOp)
    lhs = width.owner.value

    rewriter.replace_matched_op([TypeOp(IntegerType(lhs.value.data))])


def is_constant_factory(constant: int):
    def is_constant(
        op: ApplyNativeConstraintOp,
        rewriter: PatternRewriter,
        context: PDLToSMTRewriteContext,
    ) -> SSAValue:
        (value,) = op.args

        if not isinstance(value.type, smt_int.SMTIntType):
            raise Exception(
                "the constraint expects the input to be lowered to a `!smt.int.int`"
            )

        minus_one = smt_int.ConstantOp(-1)
        eq_minus_one = smt.EqOp(value, minus_one.res)
        rewriter.replace_matched_op([minus_one, eq_minus_one], [])
        return eq_minus_one.res

    return is_constant


def is_attr_equal(
    op: ApplyNativeConstraintOp,
    rewriter: PatternRewriter,
    context: PDLToSMTRewriteContext,
) -> SSAValue:
    (lhs, rhs) = op.args

    eq_op = smt.EqOp(lhs, rhs)
    rewriter.replace_matched_op([eq_op], [])
    return eq_op.res


def is_attr_not_equal(
    op: ApplyNativeConstraintOp,
    rewriter: PatternRewriter,
    context: PDLToSMTRewriteContext,
) -> SSAValue:
    (lhs, rhs) = op.args

    eq_op = smt.DistinctOp(lhs, rhs)
    rewriter.replace_matched_op([eq_op], [])
    return eq_op.res


def is_greater_integer_type(
    op: ApplyNativeConstraintOp,
    rewriter: PatternRewriter,
    context: PDLToSMTRewriteContext,
) -> SSAValue:
    (lhs_value, rhs_value) = op.args
    assert isinstance(lhs_value, ErasedSSAValue)
    assert isinstance(rhs_value, ErasedSSAValue)

    lhs_width = context.pdl_types_to_width[lhs_value.old_value]
    rhs_width = context.pdl_types_to_width[rhs_value.old_value]

    gt_op = smt_int.GtOp(lhs_width, rhs_width)
    assert_op = smt.AssertOp(gt_op.res)
    rewriter.replace_matched_op([gt_op, assert_op])

    return gt_op.res


parametric_integer_arith_native_rewrites: dict[
    str,
    Callable[[ApplyNativeRewriteOp, PatternRewriter, PDLToSMTRewriteContext], None],
] = {
    # Mathematical operations on attributes
    "addi": addi_rewrite,
    "subi": subi_rewrite,
    "muli": muli_rewrite,
    # Get constant attributes
    "get_zero_attr": get_cst_rewrite_factory(0),
    "get_one_attr": get_cst_rewrite_factory(1),
    # Integer to type conversion
    "integer_type_from_width": integer_type_from_width,
    #
}

parametric_integer_arith_native_constraints: dict[
    str,
    Callable[
        [ApplyNativeConstraintOp, PatternRewriter, PDLToSMTRewriteContext], SSAValue
    ],
] = {
    # Equality to constants
    "is_one": is_constant_factory(1),
    "is_zero": is_constant_factory(0),
    # Equality between attributes
    "is_attr_equal": is_attr_equal,
    "is_attr_not_equal": is_attr_not_equal,
    # Integer type equality
    "is_greater_integer_type": is_greater_integer_type,
}

parametric_integer_arith_native_static_constraints: dict[
    str, Callable[[ApplyNativeConstraintOp, PDLToSMTRewriteContext], bool]
] = {}
