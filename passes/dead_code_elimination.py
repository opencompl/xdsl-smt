from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext, Operation
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern
from traits.effects import Pure


class RemoveDeadPattern(RewritePattern):

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, Pure):
            return None
        if all(len(result.uses) == 0 for result in op.results):
            rewriter.erase_matched_op()


def dead_code_elimination(ctx: MLContext, module: ModuleOp):
    walker = PatternRewriteWalker(RemoveDeadPattern())
    walker.rewrite_module(module)
