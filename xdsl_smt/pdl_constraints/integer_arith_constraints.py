"""
This file contains semantics of constraints and rewrites for integer arithmetic
in PDL.
"""

from typing import Callable
from xdsl.ir import Attribute, ErasedSSAValue, Operation, SSAValue
from xdsl.utils.hints import isa
from xdsl.pattern_rewriter import PatternRewriter

from xdsl.dialects import arith
from xdsl.dialects.pdl import ApplyNativeConstraintOp, ApplyNativeRewriteOp
import xdsl_smt.dialects.smt_bitvector_dialect as smt_bv
import xdsl_smt.dialects.smt_utils_dialect as smt_utils
import xdsl_smt.dialects.smt_dialect as smt
from xdsl_smt.passes.pdl_to_smt import PDLToSMTRewriteContext


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


def invert_arith_cmpi_predicate_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    (predicate,) = op.args

    comparison_idx = {
        name: i for i, name in enumerate(arith.CMPI_COMPARISON_OPERATIONS)
    }
    replacements = {
        "eq": "ne",
        "ne": "eq",
        "slt": "sge",
        "sle": "sgt",
        "sgt": "sle",
        "sge": "slt",
        "ult": "uge",
        "ule": "ugt",
        "ugt": "ule",
        "uge": "ult",
    }
    int_replacements = {
        comparison_idx[from_]: comparison_idx[to] for from_, to in replacements.items()
    }

    next_case = predicate
    for from_, to in int_replacements.items():
        from_constant = smt_bv.ConstantOp(from_, 64)
        to_constant = smt_bv.ConstantOp(to, 64)
        eq_from = smt.EqOp(predicate, from_constant.res)
        res = smt.IteOp(eq_from.res, to_constant.res, next_case)
        next_case = res.res

        rewriter.insert_op_before_matched_op([from_constant, to_constant, eq_from, res])

    rewriter.replace_matched_op([], [next_case])


def is_constant_factory(constant: int):
    def is_constant(
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
        minus_one = smt_bv.ConstantOp(
            ((constant % 2**width) + 2**width) % 2**width, width
        )
        eq_minus_one = smt.EqOp(value, minus_one.res)
        rewriter.replace_matched_op([eq_minus_one, minus_one], [])
        return eq_minus_one.res

    return is_constant


def is_cmpi_predicate(
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
    max_predicate_int = len(arith.CMPI_COMPARISON_OPERATIONS)
    max_predicate = smt_bv.ConstantOp(max_predicate_int, width)
    predicate_valid = smt_bv.UltOp(value, max_predicate.res)

    rewriter.replace_matched_op([max_predicate, predicate_valid], [])
    return predicate_valid.res


def is_greater_integer_type(
    op: ApplyNativeConstraintOp,
    context: PDLToSMTRewriteContext,
) -> bool:
    (lhs_value, rhs_value) = op.args
    assert isinstance(lhs_value, ErasedSSAValue)
    assert isinstance(rhs_value, ErasedSSAValue)
    lhs_type = context.pdl_types_to_types[lhs_value.old_value]
    rhs_type = context.pdl_types_to_types[rhs_value.old_value]

    lhs_type = get_bv_type_from_optional_poison(lhs_type, "is_greater_integer_type")
    rhs_type = get_bv_type_from_optional_poison(rhs_type, "is_greater_integer_type")

    lhs_width = lhs_type.width.data
    rhs_width = rhs_type.width.data

    return lhs_width > rhs_width


integer_arith_native_rewrites: dict[
    str,
    Callable[[ApplyNativeRewriteOp, PatternRewriter, PDLToSMTRewriteContext], None],
] = {
    "addi": addi_rewrite,
    "subi": subi_rewrite,
    "muli": muli_rewrite,
    "get_zero_attr": get_zero_attr_rewrite,
    "invert_arith_cmpi_predicate": invert_arith_cmpi_predicate_rewrite,
}

integer_arith_native_constraints = {
    "is_minus_one": is_constant_factory(-1),
    "is_one": is_constant_factory(1),
    "is_zero": is_constant_factory(0),
    "is_arith_cmpi_predicate": is_cmpi_predicate,
}

integer_arith_native_static_constraints = {
    "is_greater_integer_type": is_greater_integer_type,
}
