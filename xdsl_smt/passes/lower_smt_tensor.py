from abc import ABC
from typing import cast, Callable
from dataclasses import dataclass
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.ir import Operation

from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_array_dialect as smt_array

from xdsl_smt.dialects.smt_dialect import BoolType, EqOp, AssertOp, DeclareFunOp, DeclareConstOp
from xdsl_smt.semantics.semantics import OperationSemantics, TypeSemantics
from xdsl.ir import Operation, SSAValue, Attribute
from typing import Mapping, Sequence
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl_smt.dialects.smt_tensor_dialect import (
    SMTTensorType,
    TensorAddOp,
    ElementwiseBinaryOperation,
    TensorTransposeOp,
    ElementwiseUnaryOperation,
    TensorSubtractOp,
)
from xdsl.dialects.builtin import ArrayAttr, FunctionType, ModuleOp, StringAttr
from xdsl.ir import Attribute
from xdsl.context import Context
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.passes import ModulePass

from ..dialects.smt_dialect import (
    CallOp,
)
from ..dialects.smt_tensor_dialect import SMTTensorType, TensorExtractOp

from .dead_code_elimination import DeadCodeElimination


def lower_tensor_type(typ: Attribute) -> Attribute:
    if isinstance(typ, SMTTensorType):
        result = typ.element_type
        index_type = smt_bv.BitVectorType(32)
        for _ in typ.shape:
            result = smt_array.ArrayType(index_type, result)
        return result
    return typ


class DeclareConstOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DeclareConstOp, rewriter: PatternRewriter):
        if isinstance(op.res.type, SMTTensorType):
            new_constant_op = DeclareConstOp(lower_tensor_type(op.res.type))
            rewriter.replace_matched_op(new_constant_op)



class TensorExtractOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorExtractOp, rewriter: PatternRewriter):
        source = op.tensor
        assert isinstance(source.type, smt_array.ArrayType)
        select_ops:list[smt_array.SelectOp] = []
        for idx in op.indices:
            select_ops.append(smt_array.SelectOp(source, idx))
            source = select_ops[-1].res
        rewriter.replace_matched_op(select_ops)


class LowerSMTTensor(ModulePass):
    name = "lower-smt-tensor"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    DeclareConstOpPattern(),
                    TensorExtractOpPattern()
                ]
            ), walk_reverse=True
        )
        walker.rewrite_module(op)

        # Apply DCE pass
        DeadCodeElimination().apply(ctx, op)