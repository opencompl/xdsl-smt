"""This file defines simple canonicalization patterns for the smt.bv dialect."""

from typing import Callable, cast, Sequence
from xdsl.ir import OpResult, SSAValue, Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
)
from xdsl_smt.dialects import smt_dialect as smt, smt_bitvector_dialect as smt_bv


def get_bv_constant(value: SSAValue) -> int | None:
    if not isinstance(value, OpResult):
        return None
    if not isinstance((constant := value.op), smt_bv.ConstantOp):
        return None
    return constant.value.value.data


def wrap_value(value: int, width: int) -> int:
    """Wrap the value around the given width."""
    max_value = 1 << width
    return ((value % max_value) + max_value) % max_value


def unsigned_to_signed(value: int, width: int) -> int:
    """Convert an unsigned value to a signed value."""
    return value - (1 << width) if value >= (1 << (width - 1)) else value


def bv_folding_canonicalization_pattern(
    op_type: type[Operation], operator: Callable[[Sequence[int]], int]
) -> type[RewritePattern]:
    class CanonicalizationPattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            # Check if the operation is of the correct type
            if not isinstance(op, op_type):
                return

            # Check if all operands are constants
            operand_cst = [get_bv_constant(operand) for operand in op.operands]
            if None in operand_cst:
                return
            operand_cst = cast(list[int], operand_cst)

            # Get the result width
            assert isinstance(type := op.results[0].type, smt_bv.BitVectorType)
            width = type.width.data

            # Perform the computation and wrap around
            value = wrap_value(operator(operand_cst), width)
            rewriter.replace_matched_op(smt_bv.ConstantOp(value, type.width))

    return CanonicalizationPattern


AddCanonicalizationPattern = bv_folding_canonicalization_pattern(
    smt_bv.AddOp, lambda operands: int.__add__(*operands)
)

SubCanonicalizationPattern = bv_folding_canonicalization_pattern(
    smt_bv.SubOp, lambda operands: int.__sub__(*operands)
)

MulCanonicalizationPattern = bv_folding_canonicalization_pattern(
    smt_bv.MulOp, lambda operands: int.__mul__(*operands)
)


def signed_comparison_folding_canonicalization_pattern(
    op_type: type[Operation], operator: Callable[[Sequence[int]], bool]
) -> type[RewritePattern]:
    class CanonicalizationPattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            # Check if the operation is of the correct type
            if not isinstance(op, op_type):
                return

            # Check if all operands are constants
            operand_cst = [get_bv_constant(operand) for operand in op.operands]
            if None in operand_cst:
                return
            operand_cst = cast(list[int], operand_cst)

            # Get the operands width
            assert isinstance(type := op.operands[0].type, smt_bv.BitVectorType)
            width = type.width.data

            # Convert the operands to signed values
            operand_cst = [
                unsigned_to_signed(operand, width) for operand in operand_cst
            ]

            # Perform the computation and wrap around
            value = operator(operand_cst)
            rewriter.replace_matched_op(smt.ConstantBoolOp(value))

    return CanonicalizationPattern


SgeCanonicalizationPattern = signed_comparison_folding_canonicalization_pattern(
    smt_bv.SgeOp, lambda operands: operands[0] >= operands[1]
)

SgtCanonicalizationPattern = signed_comparison_folding_canonicalization_pattern(
    smt_bv.SgtOp, lambda operands: operands[0] > operands[1]
)

SleCanonicalizationPattern = signed_comparison_folding_canonicalization_pattern(
    smt_bv.SleOp, lambda operands: operands[0] <= operands[1]
)

SltCanonicalizationPattern = signed_comparison_folding_canonicalization_pattern(
    smt_bv.SltOp, lambda operands: operands[0] < operands[1]
)


def unsigned_comparison_folding_canonicalization_pattern(
    op_type: type[Operation], operator: Callable[[Sequence[int]], bool]
) -> type[RewritePattern]:
    class CanonicalizationPattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            # Check if the operation is of the correct type
            if not isinstance(op, op_type):
                return

            # Check if all operands are constants
            operand_cst = [get_bv_constant(operand) for operand in op.operands]
            if None in operand_cst:
                return
            operand_cst = cast(list[int], operand_cst)

            # Perform the computation
            value = operator(operand_cst)
            rewriter.replace_matched_op(smt.ConstantBoolOp(value))

    return CanonicalizationPattern


UgeCanonicalizationPattern = unsigned_comparison_folding_canonicalization_pattern(
    smt_bv.UgeOp, lambda operands: operands[0] >= operands[1]
)

UgtCanonicalizationPattern = unsigned_comparison_folding_canonicalization_pattern(
    smt_bv.UgtOp, lambda operands: operands[0] > operands[1]
)

UleCanonicalizationPattern = unsigned_comparison_folding_canonicalization_pattern(
    smt_bv.UleOp, lambda operands: operands[0] <= operands[1]
)

UltCanonicalizationPattern = unsigned_comparison_folding_canonicalization_pattern(
    smt_bv.UltOp, lambda operands: operands[0] < operands[1]
)
