from xdsl.pattern_rewriter import (
    RewritePattern,
    op_type_rewrite_pattern,
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.passes import ModulePass
from xdsl.ir import MLContext
from xdsl.dialects.builtin import IntegerType, ModuleOp

from .arith_to_smt import FuncToSMTPattern, ReturnPattern
from dialects import comb
import dialects.smt_bitvector_dialect as bv_dialect


class AddRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: comb.AddOp, rewriter: PatternRewriter):
        assert isinstance(op.result.typ, IntegerType)
        width = op.result.typ.width.data
        if len(op.operands) == 0:
            rewriter.replace_matched_op(bv_dialect.ConstantOp(0, width))
            return

        current_val = op.operands[0]

        for operand in op.operands[1:]:
            new_op = bv_dialect.AddOp(current_val, operand)
            current_val = new_op.res
            rewriter.insert_op_before_matched_op(new_op)

        rewriter.replace_matched_op([], [current_val])


comb_to_smt_patterns: list[RewritePattern] = [
    AddRewritePattern(),
]


class CombToSMT(ModulePass):
    name = "comb-to-smt"

    def apply(self, ctx: MLContext, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [*comb_to_smt_patterns, FuncToSMTPattern(), ReturnPattern()]
            )
        )
        walker.rewrite_module(op)
