from dataclasses import dataclass

from xdsl.passes import ModulePass
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, FunctionType
from xdsl.dialects import llvm, func
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)


class FuncPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.FuncOp, rewriter: PatternRewriter) -> None:
        region = op.detach_region(0)
        function_type = FunctionType.from_lists(
            op.function_type.inputs.data, [op.function_type.output]
        )
        rewriter.replace_matched_op(
            func.FuncOp(op.sym_name.data, function_type, region, op.sym_visibility)
        )


class ReturnPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.ReturnOp, rewriter: PatternRewriter) -> None:
        if not op.arg:
            rewriter.erase_matched_op()
            return
        rewriter.replace_matched_op(func.ReturnOp(op.arg))


@dataclass(frozen=True)
class RaiseLLVMToFunc(ModulePass):
    name = "raise-llvm-to-func"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier([FuncPattern(), ReturnPattern()])
        ).rewrite_module(op)
