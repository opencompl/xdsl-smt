from typing import cast
from dataclasses import dataclass
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.ir import Operation

from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import transfer
from xdsl_smt.passes.lower_to_smt.smt_lowerer import (
    SMTLowerer,
)
from xdsl_smt.dialects.smt_utils_dialect import (
    AnyPairType,
    PairType,
    SecondOp,
    FirstOp,
    PairOp,
)
from xdsl_smt.dialects.smt_dialect import BoolType, EqOp, AssertOp
from xdsl_smt.semantics.semantics import OperationSemantics, TypeSemantics
from xdsl.ir import Operation, SSAValue, Attribute
from typing import Mapping, Sequence
from xdsl.utils.isattr import isattr
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl_smt.dialects.smt_tensor_dialect import (
    SMTTensorType,
    TensorAddOp,
    ElementwiseBinaryOperation,
    TensorTransposeOp,
    ElementwiseUnaryOperation,
)
from xdsl.dialects.builtin import ArrayAttr, FunctionType, ModuleOp, StringAttr
from xdsl.ir import Attribute
from xdsl.context import MLContext
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import Rewriter, InsertPoint
from xdsl.passes import ModulePass
from xdsl.utils.hints import isa

from ..dialects.smt_dialect import (
    CallOp,
    DeclareConstOp,
    DefineFunOp,
    ReturnOp,
    ForallOp,
    EqOp,
    DistinctOp,
    OrOp,
    AndOp,
)
from ..dialects.smt_tensor_dialect import SMTTensorType
from ..dialects.smt_utils_dialect import (
    AnyPairType,
    FirstOp,
    PairOp,
    PairType,
    SecondOp,
)
from .dead_code_elimination import DeadCodeElimination


def lower_tensor_type_to_dimensions(type: Attribute) -> Attribute:
    if isinstance(type, SMTTensorType):
        result = BoolType()
        bv_type = smt_bv.BitVectorType(32)
        for _ in type.shape:
            result = PairType(bv_type, result)
        return result
    return type


class RemoveTensorArgsFunction(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DefineFunOp, rewriter: PatternRewriter):
        block = op.body.blocks[0]
        i = 0
        while i < len(block.args):
            arg = block.args[i]
            if isa(typ := arg.type, SMTTensorType):
                dimension_arg = rewriter.insert_block_argument(
                    block, i + 1, lower_tensor_type_to_dimensions(typ)
                )
                arg.replace_by(dimension_arg)
                rewriter.erase_block_argument(arg)
            else:
                i += 1
        old_typ = op.ret.type
        assert isinstance(old_typ, FunctionType)
        new_inputs = [arg.type for arg in block.args]
        new_outputs = [lower_tensor_type_to_dimensions(ty) for ty in old_typ.outputs]
        op.ret.type = FunctionType.from_attrs(
            ArrayAttr[Attribute](new_inputs), ArrayAttr[Attribute](new_outputs)
        )


class LowerElementwiseBinaryOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ElementwiseBinaryOperation, rewriter: PatternRewriter
    ):
        if not isa((pair_type := op.result.type), AnyPairType):
            eq_op = EqOp(op.lhs, op.rhs)
            rewriter.insert_op_before_matched_op([eq_op, AssertOp(eq_op.res)])
            op.result.replace_by(op.lhs)
            rewriter.erase_matched_op()


class LowerElementwiseUnaryOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ElementwiseUnaryOperation, rewriter: PatternRewriter
    ):
        if not isa((pair_type := op.result.type), AnyPairType):
            eq_op = EqOp(op.op)
            rewriter.insert_op_before_matched_op([eq_op, AssertOp(eq_op.res)])
            op.result.replace_by(op.op)
            rewriter.erase_matched_op()


class LowerTransposeOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorTransposeOp, rewriter: PatternRewriter):
        if not isa((pair_type := op.result.type), AnyPairType):
            cur_val = op.operands[0]
            ops = []
            dim_list = []
            for _ in op.permutation.get_values():
                ops.append(FirstOp(cur_val))
                dim_list.append(ops[-1].res)
                ops.append(SecondOp(cur_val))
                cur_val = ops[-1].res
            new_dim_list = []
            for i in op.permutation.get_values():
                new_dim_list.append(dim_list[i])
            for val in new_dim_list[::-1]:
                ops.append(PairOp(val, cur_val))
                cur_val = ops[-1].results[0]
            rewriter.insert_op_before_matched_op(ops)
            op.result.replace_by(cur_val)
            rewriter.erase_matched_op()


class TestOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        print(op)


class LowerTensorToDimensions(ModulePass):
    name = "lower-tensor-to-dimensions"

    def apply(self, ctx: MLContext, op: ModuleOp):
        # Remove pairs from function arguments.
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemoveTensorArgsFunction(),
                    LowerElementwiseBinaryOpPattern(),
                    LowerTransposeOpPattern(),
                ]
                # [TestOpPattern()]
            )
        )
        walker.rewrite_module(op)

        # Apply DCE pass
        DeadCodeElimination().apply(ctx, op)
