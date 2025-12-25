from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)

from xdsl.ir import SSAValue
from xdsl_smt.dialects.smt_tensor_dialect import (
    TensorTransposeOp,
    TensorExtractOp,
)
from xdsl.dialects.builtin import ModuleOp
from xdsl.context import Context
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.passes import ModulePass


class RewriteTransposeOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorTransposeOp, rewriter: PatternRewriter):
        for use in op.result.uses:
            extract_op = use.operation
            if isinstance(extract_op, TensorExtractOp):
                permutations = op.get_permutation()
                new_indices: list[SSAValue] = []
                for i in permutations:
                    new_indices.append(extract_op.indices[i])
                new_extract_op = TensorExtractOp(op.operand, new_indices)
                rewriter.replace_op(extract_op, new_extract_op)
        if op.result.uses.get_length() == 0:
            rewriter.erase_matched_op()


class RewriteSMTTensor(ModulePass):
    """
    Rewrite patterns like `extract(op(arg))` to `extract(arg')`
    """

    name = "rewrite-smt-tensor"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([RewriteTransposeOpPattern()])
        )
        walker.rewrite_module(op)
        CommonSubexpressionElimination().apply(ctx, op)
