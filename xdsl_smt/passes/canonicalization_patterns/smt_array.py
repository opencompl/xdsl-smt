from xdsl.ir import OpResult
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl_smt.dialects import smt_array_dialect as array


class SelectStorePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: array.SelectOp, rewriter: PatternRewriter):
        # select id (store id x) -> x
        if not isinstance(store := op.array.owner, array.StoreOp):
            return None
        if op.index != store.index:
            return None
        rewriter.replace_matched_op([], [store.value])
