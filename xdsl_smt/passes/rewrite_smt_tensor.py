from abc import ABC
from typing import cast, Callable
from dataclasses import dataclass
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.ir import Operation

from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_array_dialect as smt_array

from xdsl_smt.dialects.smt_dialect import BoolType, EqOp, AssertOp, DeclareFunOp
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


class TensorRewritePattern(RewritePattern, ABC):
    extract_op:TensorExtractOp

    def __init__(self, extract_op):
        self.extract_op = extract_op
        super().__init__()


class LowerTransposeOpPattern(TensorRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorTransposeOp, rewriter: PatternRewriter):
        extract_op = self.extract_op
        permutations = op.permutation.get_values()
        new_indices:list[SSAValue] = []
        for i in permutations:
            new_indices.append(extract_op.indices[i])
        new_extract_op = TensorExtractOp(op.operand, new_indices)
        rewriter.replace_op(extract_op, new_extract_op)
        rewriter.erase_matched_op()

def stripOpName(name:str) -> str:
    assert '.' in name
    return name[name.index(".")+1:]

elementwise_unary_function_set:set[DeclareFunOp] = set()
elementwise_unary_functions:dict[str, Callable[[SSAValue],CallOp]] = {}


elementwise_binary_function_set:set[DeclareFunOp] = set()
elementwise_binary_functions:dict[str, Callable[[SSAValue, SSAValue],CallOp]] = {}


def getElementwiseFunction(op_name:str, element_type:Attribute, function_set:set[DeclareFunOp],
                                  function_dict:dict[str, Callable]):
    if op_name not in function_dict:
        element_uf_type = FunctionType.from_lists([element_type, element_type], [element_type])
        defun_op = DeclareFunOp(element_uf_type, op_name)
        function_set.add(defun_op)
        function_dict[op_name] = lambda x, y: CallOp(defun_op.ret, [x, y])
    return function_dict[op_name]


class LowerElementwiseUnaryOpPattern(TensorRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ElementwiseUnaryOperation, rewriter: PatternRewriter
    ):
        element_type = self.extract_op.result.type
        op_name = stripOpName(op.name)
        unary_function = getElementwiseFunction(op_name, element_type,
                                                 elementwise_unary_function_set, elementwise_unary_functions)
        extract_op_op = TensorExtractOp(op.op, self.extract_op.indices)
        call_op = unary_function(extract_op_op.result)
        rewriter.replace_op(self.extract_op, [extract_op_op, call_op])
        rewriter.erase_matched_op()


class LowerElementwiseBinaryOpPattern(TensorRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ElementwiseBinaryOperation, rewriter: PatternRewriter
    ):
        element_type = self.extract_op.result.type
        op_name = stripOpName(op.name)
        binary_function = getElementwiseFunction(op_name, element_type,
                                                 elementwise_binary_function_set, elementwise_binary_functions)
        extract_lhs_op = TensorExtractOp(op.lhs, self.extract_op.indices)
        extract_rhs_op = TensorExtractOp(op.rhs, self.extract_op.indices)
        call_op = binary_function(extract_lhs_op.result, extract_rhs_op.result)
        rewriter.replace_op(self.extract_op, [extract_lhs_op, extract_rhs_op, call_op])
        rewriter.erase_matched_op()



class TestOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        print(op)



class TensorExtractOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorExtractOp, rewriter: PatternRewriter):
        source = op.tensor
        source_parent_op = source.owner
        if isinstance(source_parent_op, ElementwiseBinaryOperation):
            ...
        elif isinstance(source_parent_op, ElementwiseUnaryOperation):
            LowerElementwiseBinaryOpPattern(source_parent_op).match_and_rewrite(source_parent_op, rewriter)
        elif isinstance(source_parent_op, TensorTransposeOp):
            LowerTransposeOpPattern(op).match_and_rewrite(source_parent_op, rewriter)



class InsertFunctionPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter):
        block = op.body.block
        first_op = block.first_op
        assert first_op is not None

        while len(elementwise_binary_function_set) > 0:
            function_op = elementwise_binary_function_set.pop()
            block.insert_op_before(function_op, first_op)

        while len(elementwise_unary_function_set) > 0:
            function_op = elementwise_unary_function_set.pop()
            block.insert_op_before(function_op, first_op)



class LowerSMTTensor(ModulePass):
    name = "rewrite-smt-tensor"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    TensorExtractOpPattern()
                ]
            ), walk_reverse=True
        )
        walker.rewrite_module(op)

        walker1 = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    InsertFunctionPattern()
                ]
            ), apply_recursively=False
        )
        walker1.rewrite_module(op)