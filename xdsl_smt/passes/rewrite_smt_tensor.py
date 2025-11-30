from abc import ABC


from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv

from xdsl.ir import SSAValue
from xdsl_smt.dialects.smt_tensor_dialect import (
    TensorTransposeOp,
    INDEX_WIDTH,
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


bv_constants: dict[int, smt_bv.ConstantOp] = {}


def getBVConstant(x: int) -> smt_bv.ConstantOp:
    global bv_constants
    if x not in bv_constants:
        bv_constants[x] = smt_bv.ConstantOp.from_int_value(x, INDEX_WIDTH)
    return bv_constants[x]


class TensorRewritePattern(RewritePattern, ABC):
    extract_op: TensorExtractOp

    def __init__(self, extract_op: TensorExtractOp):
        self.extract_op = extract_op
        super().__init__()


class RewriteTransposeOpPattern(TensorRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorTransposeOp, rewriter: PatternRewriter):
        extract_op = self.extract_op
        permutations = op.get_permutation()
        new_indices: list[SSAValue] = []
        for i in permutations:
            new_indices.append(extract_op.indices[i])
        new_extract_op = TensorExtractOp(op.operand, new_indices)
        rewriter.replace_op(extract_op, new_extract_op)
        rewriter.erase_matched_op()


class TensorExtractOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorExtractOp, rewriter: PatternRewriter):
        source = op.tensor
        source_parent_op = source.owner
        if isinstance(source_parent_op, TensorTransposeOp):
            RewriteTransposeOpPattern(op).match_and_rewrite(source_parent_op, rewriter)


def insertConstantsBeforeModule(op: ModuleOp):
    global bv_constants

    block = op.body.block
    first_op = block.first_op
    assert first_op is not None
    for val in bv_constants.values():
        block.insert_op_before(val, first_op)


class RewriteSMTTensor(ModulePass):
    name = "rewrite-smt-tensor"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([TensorExtractOpPattern()]), walk_reverse=True
        )
        walker.rewrite_module(op)

        insertConstantsBeforeModule(op)
