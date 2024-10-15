from xdsl.ir import OpResult
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl_smt.dialects import (
    smt_utils_dialect as smt_utils,
)


class FirstCanonicalizationPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: smt_utils.FirstOp, rewriter: PatternRewriter):
        # first (pair x y) -> x
        if not isinstance(op.pair, OpResult):
            return None
        if not isinstance(op.pair.op, smt_utils.PairOp):
            return None
        rewriter.replace_matched_op([], [op.pair.op.first])
        return


class SecondCanonicalizationPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: smt_utils.SecondOp, rewriter: PatternRewriter):
        # second (pair x y) -> y
        if not isinstance(op.pair, OpResult):
            return None
        if not isinstance(op.pair.op, smt_utils.PairOp):
            return None
        rewriter.replace_matched_op([], [op.pair.op.second])
        return
