from dataclasses import dataclass

from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp, FunctionType
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from xdsl_smt.dialects.smt_dialect import DefineFunOp, ReturnOp
from xdsl_smt.dialects.smt_utils_dialect import (
    pair_type_from_list,
    merge_values_with_pairs,
)


class LowerFunctionPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DefineFunOp, rewriter: PatternRewriter):
        func_type = op.ret.type
        assert isinstance(func_type, FunctionType)

        if len(func_type.outputs) == 1:
            return

        new_type = FunctionType.from_lists(
            func_type.inputs.data, [pair_type_from_list(*func_type.outputs.data)]
        )
        rewriter.replace_value_with_new_type(op.ret, new_type)


class ReturnPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter):
        if len(op.ret) == 1:
            return
        new_result = merge_values_with_pairs(op.ret, rewriter, InsertPoint.before(op))
        rewriter.replace_matched_op(ReturnOp(new_result))


@dataclass(frozen=True)
class MergeFuncResultsPass(ModulePass):
    """
    Merge the results of a function into nested pairs.
    This is necessary to convert to the SMT-LIB format, as SMT-LIB does not support
    multiple return values from functions.
    """

    name = "merge-func-results"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerFunctionPattern(), ReturnPattern()])
        )
        walker.rewrite_module(op)
