from typing import Sequence
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.utils.hints import isa

from ...dialects import smt_dialect
from ...dialects import smt_bitvector_dialect as bv_dialect
from ...dialects import arith_dialect as arith
from ...dialects import smt_utils_dialect as utils_dialect


def get_int_value_and_poison(
    val: SSAValue, rewriter: PatternRewriter
) -> tuple[SSAValue, SSAValue]:
    value = utils_dialect.FirstOp(val)
    poison = utils_dialect.SecondOp(val)
    rewriter.insert_op_before_matched_op([value, poison])
    return value.res, poison.res


class IntegerConstantRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter):
        if not isa(op.value, IntegerAttr[IntegerType]):
            raise Exception("Cannot convert constant of type that are not integer type")
        value_op = bv_dialect.ConstantOp(op.value)
        poison_op = smt_dialect.ConstantBoolOp.from_bool(False)
        res_op = utils_dialect.PairOp(value_op.res, poison_op.res)
        rewriter.replace_matched_op([value_op, poison_op, res_op])


def reduce_poison_values(
    operands: Sequence[SSAValue], rewriter: PatternRewriter
) -> tuple[Sequence[SSAValue], SSAValue]:
    assert len(operands) == 2

    left_value, left_poison = get_int_value_and_poison(operands[0], rewriter)
    right_value, right_poison = get_int_value_and_poison(operands[1], rewriter)
    res_poison_op = smt_dialect.OrOp(left_poison, right_poison)
    rewriter.insert_op_before_matched_op(res_poison_op)
    return [left_value, right_value], res_poison_op.res


class AddiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.AddOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class AndiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Andi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.AndOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class OriRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Ori, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.OrOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


arith_to_smt_patterns: list[RewritePattern] = [
    IntegerConstantRewritePattern(),
    AddiRewritePattern(),
    AndiRewritePattern(),
    OriRewritePattern(),
]
