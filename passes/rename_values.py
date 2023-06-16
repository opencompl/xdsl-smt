from xdsl.dialects.func import FuncOp
from dataclasses import dataclass
from xdsl.passes import ModulePass

from xdsl.ir import Operation, MLContext

from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.dialects import builtin


@dataclass
class RenameOpResult(RewritePattern):
    autogen = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if isinstance(op, FuncOp):
            for arg in op.args:
                if arg.name_hint is None:
                    arg.name_hint = "autogen_arg" + str(self.autogen)
                    self.autogen += 1
        if not (isinstance(op, builtin.ModuleOp) or isinstance(op, FuncOp)):
            if len(op.results) > 0:
                for res in op.results:
                    if res.name_hint is None:
                        res.name_hint = "autogen" + str(self.autogen)
                        self.autogen += 1


@dataclass
class RenameValuesPass(ModulePass):
    name = "rename_op_result"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([RenameOpResult()]),
            walk_regions_first=True,
            apply_recursively=True,
            walk_reverse=False,
        )
        walker.rewrite_module(op)
