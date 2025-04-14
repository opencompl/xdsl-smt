from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.context import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern
from ..traits.effects import Pure


class RemoveDeadPattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, Pure):
            return None
        if all(len(result.uses) == 0 for result in op.results):
            rewriter.erase_matched_op()


class DeadCodeElimination(ModulePass):
    """
    Defines a simple dead code elimination pass.
    It eliminates every operation that is not used and has no side effects.
    """

    name = "dce"

    def apply(self, ctx: MLContext, op: ModuleOp):
        walker = PatternRewriteWalker(RemoveDeadPattern(), walk_reverse=True)
        walker.rewrite_module(op)
