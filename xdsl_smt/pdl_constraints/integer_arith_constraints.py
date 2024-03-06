"""
This file contains semantics of constraints and rewrites for integer arithmetic
in PDL.
"""

from typing import Callable
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import Attribute, ErasedSSAValue, Operation, SSAValue
from xdsl.utils.hints import isa
from xdsl.pattern_rewriter import PatternRewriter

from xdsl.dialects import arith
from xdsl.dialects import comb
from xdsl.dialects.pdl import (
    ApplyNativeConstraintOp,
    ApplyNativeRewriteOp,
    AttributeOp,
    TypeOp,
)
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


def andi_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    return single_op_rewrite(op, rewriter, smt_bv.AndOp)


def ori_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    return single_op_rewrite(op, rewriter, smt_bv.OrOp)


def xori_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    return single_op_rewrite(op, rewriter, smt_bv.XorOp)


def shl_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    return single_op_rewrite(op, rewriter, smt_bv.ShlOp)


def get_cst_rewrite_factory(constant: int):
    def get_cst_rewrite(
        op: ApplyNativeRewriteOp,
        rewriter: PatternRewriter,
        context: PDLToSMTRewriteContext,
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

        zero = smt_bv.ConstantOp(
            ((constant % (1 << width)) + (1 << width)) % (1 << width), width
        )
        rewriter.replace_matched_op([zero])

    return get_cst_rewrite


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


def invert_comb_icmp_predicate_rewrite(
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


def integer_type_sub_width(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    (lhs, rhs) = op.args
    assert isinstance(lhs, ErasedSSAValue)
    lhs_type = context.pdl_types_to_types[lhs.old_value]
    assert isa(lhs_type, smt_utils.PairType[smt_bv.BitVectorType, smt.BoolType])

    assert isinstance(rhs, ErasedSSAValue)
    rhs_type = context.pdl_types_to_types[rhs.old_value]
    assert isa(rhs_type, smt_utils.PairType[smt_bv.BitVectorType, smt.BoolType])

    type_op = TypeOp(IntegerType(lhs_type.first.width.data - rhs_type.first.width.data))
    rewriter.replace_matched_op([type_op])


def integer_type_add_width(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    (lhs, rhs) = op.args
    assert isinstance(lhs, ErasedSSAValue)
    lhs_type = context.pdl_types_to_types[lhs.old_value]
    assert isa(lhs_type, smt_utils.PairType[smt_bv.BitVectorType, smt.BoolType])

    assert isinstance(rhs, ErasedSSAValue)
    rhs_type = context.pdl_types_to_types[rhs.old_value]
    assert isa(rhs_type, smt_utils.PairType[smt_bv.BitVectorType, smt.BoolType])

    type_op = TypeOp(IntegerType(lhs_type.first.width.data + rhs_type.first.width.data))
    rewriter.replace_matched_op([type_op])


def cast_to_type_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    lhs, rhs = op.args
    assert isinstance(rhs, ErasedSSAValue)
    rhs_type = context.pdl_types_to_types[rhs.old_value]
    assert isa(rhs_type, smt_utils.PairType[smt_bv.BitVectorType, smt.BoolType])

    assert isinstance(lhs.owner, smt_bv.ConstantOp)
    lhs = lhs.owner.value

    attribute_op = AttributeOp(IntegerAttr(lhs.value.data, rhs_type.first.width.data))
    rewriter.replace_matched_op([attribute_op])


def is_constant_factory(constant: int):
    def is_constant(
        op: ApplyNativeConstraintOp,
        rewriter: PatternRewriter,
        context: PDLToSMTRewriteContext,
    ) -> SSAValue:
        (value,) = op.args

        if not isinstance(value.type, smt_bv.BitVectorType):
            raise Exception(
                "the constraint expects the input to be lowered to a `!smt.bv<...>`"
            )

        width = value.type.width.data
        minus_one = smt_bv.ConstantOp(
            ((constant % 2**width) + 2**width) % 2**width, width
        )
        eq_minus_one = smt.EqOp(value, minus_one.res)
        rewriter.replace_matched_op([eq_minus_one, minus_one], [])
        return eq_minus_one.res

    return is_constant


def get_constant_factory(constant: int):
    def get_constant(
        op: ApplyNativeRewriteOp,
        rewriter: PatternRewriter,
        context: PDLToSMTRewriteContext,
    ) -> None:
        (type,) = op.args

        assert isinstance(type, ErasedSSAValue)
        type = context.pdl_types_to_types[type.old_value]
        assert isa(type, smt_utils.PairType[smt_bv.BitVectorType, smt.BoolType])

        width = type.first.width.data
        attr_op = AttributeOp(IntegerAttr(constant, width))
        rewriter.replace_matched_op([attr_op])

    return get_constant


def get_width(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, context: PDLToSMTRewriteContext
) -> None:
    (type_with_width, expected_type) = op.args
    assert isinstance(type_with_width, ErasedSSAValue)
    type_with_width = context.pdl_types_to_types[type_with_width.old_value]
    type_with_width = get_bv_type_from_optional_poison(type_with_width, "get_width")
    assert isinstance(expected_type, ErasedSSAValue)
    expected_type = context.pdl_types_to_types[expected_type.old_value]
    expected_type = get_bv_type_from_optional_poison(expected_type, "get_width")

    attr_op = AttributeOp(
        IntegerAttr(type_with_width.width.data, expected_type.width.data)
    )
    rewriter.replace_matched_op([attr_op])


def is_not_zero(
    op: ApplyNativeConstraintOp,
    rewriter: PatternRewriter,
    context: PDLToSMTRewriteContext,
) -> SSAValue:
    (value,) = op.args

    if not isinstance(value.type, smt_bv.BitVectorType):
        raise Exception(
            "is_not_zero expects the input to be lowered to a `!smt.bv<...>`"
        )

    width = value.type.width.data
    zero = smt_bv.ConstantOp(0, width)
    ne_zero = smt.DistinctOp(value, zero.res)
    rewriter.replace_matched_op([zero, ne_zero], [])
    return ne_zero.res


def is_attr_not_equal(
    op: ApplyNativeConstraintOp,
    rewriter: PatternRewriter,
    context: PDLToSMTRewriteContext,
) -> SSAValue:
    (lhs, rhs) = op.args

    eq_op = smt.DistinctOp(lhs, rhs)
    rewriter.replace_matched_op([eq_op], [])
    return eq_op.res


def is_arith_cmpi_predicate(
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


def is_comb_icmp_predicate(
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
    max_predicate_int = len(comb.ICMP_COMPARISON_OPERATIONS)
    max_predicate = smt_bv.ConstantOp(max_predicate_int, width)
    predicate_valid = smt_bv.UltOp(value, max_predicate.res)

    rewriter.replace_matched_op([max_predicate, predicate_valid], [])
    return predicate_valid.res


def truncation_match_shift_amount(
    op: ApplyNativeConstraintOp,
    rewriter: PatternRewriter,
    context: PDLToSMTRewriteContext,
) -> SSAValue:
    """
    Check that the truncation amount from type arg[0] to type arg[1] is equal
    to the the value of the arg[2] attribute.
    """
    (previous_type, new_type, shift_amount) = op.args

    assert isinstance(previous_type, ErasedSSAValue)
    previous_type = context.pdl_types_to_types[previous_type.old_value]
    assert isinstance(new_type, ErasedSSAValue)
    new_type = context.pdl_types_to_types[new_type.old_value]

    previous_type = get_bv_type_from_optional_poison(
        previous_type, "truncation_match_shift_amount"
    )
    new_type = get_bv_type_from_optional_poison(
        new_type, "truncation_match_shift_amount"
    )

    trunc_amonut = previous_type.width.data - new_type.width.data
    trunc_amount_constant = smt_bv.ConstantOp(trunc_amonut, previous_type.width.data)
    eq_trunc_amount = smt.EqOp(shift_amount, trunc_amount_constant.res)

    rewriter.replace_matched_op([eq_trunc_amount, trunc_amount_constant], [])
    return eq_trunc_amount.res


def is_equal_to_width_of_type(
    op: ApplyNativeConstraintOp,
    rewriter: PatternRewriter,
    context: PDLToSMTRewriteContext,
) -> SSAValue:
    (width_value, erased_int_type) = op.args
    assert isinstance(width_value.type, smt_bv.BitVectorType)
    assert isinstance(erased_int_type, ErasedSSAValue)
    pair_int_type = context.pdl_types_to_types[erased_int_type.old_value]
    int_type = get_bv_type_from_optional_poison(
        pair_int_type, "is_equal_to_width_of_type"
    )

    # If we cannot even put the width of the type in the width_value attribute,
    # then the result has to be False.
    if int_type.width.data >= 2**width_value.type.width.data:
        false_op = smt.ConstantBoolOp(False)
        rewriter.replace_matched_op([false_op], [])
        return false_op.res

    width_op = smt_bv.ConstantOp(int_type.width, width_value.type.width)
    eq_width_op = smt.EqOp(width_value, width_op.res)
    rewriter.replace_matched_op([eq_width_op, width_op], [])

    return eq_width_op.res


def get_minimum_signed_value(
    op: ApplyNativeRewriteOp,
    rewriter: PatternRewriter,
    context: PDLToSMTRewriteContext,
) -> None:
    (type,) = op.args
    assert isinstance(type, ErasedSSAValue)
    type = context.pdl_types_to_types[type.old_value]
    type = get_bv_type_from_optional_poison(type, "get_minimum_signed_value")

    width = type.width.data

    attr_op = AttributeOp(IntegerAttr(2 ** (width - 1), width))
    rewriter.replace_matched_op([attr_op])


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
    "andi": andi_rewrite,
    "ori": ori_rewrite,
    "xori": xori_rewrite,
    "shl": shl_rewrite,
    "get_zero_attr": get_cst_rewrite_factory(0),
    "get_one_attr": get_cst_rewrite_factory(1),
    "get_minus_one_attr": get_cst_rewrite_factory(-1),
    "invert_arith_cmpi_predicate": invert_arith_cmpi_predicate_rewrite,
    "invert_comb_icmp_predicate": invert_comb_icmp_predicate_rewrite,
    "integer_type_sub_width": integer_type_sub_width,
    "integer_type_add_width": integer_type_add_width,
    "cast_to_type": cast_to_type_rewrite,
    "get_one": get_constant_factory(1),
    "get_width": get_width,
    "get_minimum_signed_value": get_minimum_signed_value,
}

integer_arith_native_constraints = {
    "is_minus_one": is_constant_factory(-1),
    "is_one": is_constant_factory(1),
    "is_zero": is_constant_factory(0),
    "is_not_zero": is_not_zero,
    "is_attr_not_equal": is_attr_not_equal,
    "is_arith_cmpi_predicate": is_arith_cmpi_predicate,
    "is_comb_icmp_predicate": is_comb_icmp_predicate,
    "truncation_match_shift_amount": truncation_match_shift_amount,
    "is_equal_to_width_of_type": is_equal_to_width_of_type,
}

integer_arith_native_static_constraints = {
    "is_greater_integer_type": is_greater_integer_type,
}
