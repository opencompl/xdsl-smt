from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern
from xdsl_smt.traits.effects import Pure
from xdsl import traits


class RemoveDeadPattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, Pure) and not op.has_trait(traits.Pure):
            return None
        if all(not result.uses for result in op.results):
            rewriter.erase_matched_op()


class DeadCodeElimination(ModulePass):
    name = "dce"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(RemoveDeadPattern(), walk_reverse=True)
        walker.rewrite_module(op)
