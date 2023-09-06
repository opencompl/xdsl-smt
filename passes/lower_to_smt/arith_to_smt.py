from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.utils.hints import isa

import dialects.smt_bitvector_dialect as bv_dialect
import dialects.arith_dialect as arith


class IntegerConstantRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter):
        if not isa(op.value, IntegerAttr[IntegerType]):
            raise Exception("Cannot convert constant of type that are not integer type")
        smt_op = bv_dialect.ConstantOp(op.value)
        rewriter.replace_matched_op(smt_op)


class AddiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter):
        smt_op = bv_dialect.AddOp(op.lhs, op.rhs)
        rewriter.replace_matched_op(smt_op)


class AndiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Andi, rewriter: PatternRewriter):
        smt_op = bv_dialect.AndOp(op.lhs, op.rhs)
        rewriter.replace_matched_op(smt_op)


class OriRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Ori, rewriter: PatternRewriter):
        smt_op = bv_dialect.OrOp(op.lhs, op.rhs)
        rewriter.replace_matched_op(smt_op)


arith_to_smt_patterns: list[RewritePattern] = [
    IntegerConstantRewritePattern(),
    AddiRewritePattern(),
    AndiRewritePattern(),
    OriRewritePattern(),
]
