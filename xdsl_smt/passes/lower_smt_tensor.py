from xdsl_smt.dialects import smt_array_dialect as smt_array

from xdsl_smt.dialects.smt_dialect import (
    DeclareConstOp,
)
from xdsl_smt.dialects.smt_tensor_dialect import (
    IndexType,
    SMTTensorType,
    TensorExtractOp,
)
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute
from xdsl.utils.hints import isa
from xdsl.context import Context
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.passes import ModulePass


def lower_tensor_type(typ: Attribute) -> Attribute:
    if isa(typ, SMTTensorType):
        result = typ.element_type
        index_type = IndexType
        for _ in typ.shape:
            result = smt_array.ArrayType(index_type, result)
        return result
    return typ


class DeclareConstOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DeclareConstOp, rewriter: PatternRewriter):
        if isa(op.res.type, SMTTensorType):
            new_constant_op = DeclareConstOp(lower_tensor_type(op.res.type))
            rewriter.replace_matched_op(new_constant_op)


class TensorExtractOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorExtractOp, rewriter: PatternRewriter):
        source = op.tensor
        assert isinstance(source.type, smt_array.ArrayType)
        select_ops: list[smt_array.SelectOp] = []
        for idx in op.indices:
            select_ops.append(smt_array.SelectOp(source, idx))
            source = select_ops[-1].res
        rewriter.replace_matched_op(select_ops)


class LowerSMTTensor(ModulePass):
    name = "lower-smt-tensor"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [DeclareConstOpPattern(), TensorExtractOpPattern()]
            )
        )
        walker.rewrite_module(op)
