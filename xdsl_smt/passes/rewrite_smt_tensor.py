from abc import ABC
from typing import cast, Callable
from dataclasses import dataclass

from xdsl.dialects.smt import AndOp
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.ir import Operation

from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_array_dialect as smt_array

from xdsl_smt.dialects.smt_dialect import BoolType, EqOp, AssertOp, DeclareFunOp, IteOp
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
    TensorSubtractOp, TensorPadOp, INDEX_WIDTH, toTupleInt, TensorSliceOp,
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

bv_constants:dict[int, smt_bv.ConstantOp] = {}

def getBVConstant(x:int) -> smt_bv.ConstantOp:
    if x not in bv_constants:
        bv_constants[x]=smt_bv.ConstantOp.from_int_value(x, INDEX_WIDTH)
    return bv_constants[x]

class TensorRewritePattern(RewritePattern, ABC):
    extract_op:TensorExtractOp

    def __init__(self, extract_op):
        self.extract_op = extract_op
        super().__init__()


class RewriteTransposeOpPattern(TensorRewritePattern):
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
    return name.replace(".", "_")


elementwise_unary_function_set:set[DeclareFunOp] = set()
elementwise_unary_functions:dict[str, Callable[[SSAValue],Operation]] = {}


elementwise_binary_function_set:set[DeclareFunOp] = set()
elementwise_binary_functions:dict[str, Callable[[SSAValue, SSAValue],Operation]] = {}


def initElementwiseIntFunction():
    global elementwise_binary_functions
    elementwise_binary_functions["smt_tensor_add"] = lambda x, y: smt_bv.AddOp(x, y)
    elementwise_binary_functions["smt_tensor_subtract"] = lambda x, y: smt_bv.SubOp(x, y)

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
        global elementwise_unary_function_set
        global elementwise_unary_functions
        element_type = self.extract_op.result.type
        op_name = stripOpName(op.name)
        unary_function = getElementwiseFunction(op_name, element_type,
                                                 elementwise_unary_function_set, elementwise_unary_functions)
        extract_op_op = TensorExtractOp(op.op, self.extract_op.indices)
        call_op = unary_function(extract_op_op.result)
        rewriter.replace_op(self.extract_op, [extract_op_op, call_op])
        rewriter.erase_matched_op()


class RewriteElementwiseBinaryOpPattern(TensorRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ElementwiseBinaryOperation, rewriter: PatternRewriter
    ):
        global elementwise_binary_function_set
        global elementwise_binary_functions
        element_type = self.extract_op.result.type
        op_name = stripOpName(op.name)
        binary_function = getElementwiseFunction(op_name, element_type,
                                                 elementwise_binary_function_set, elementwise_binary_functions)
        extract_lhs_op = TensorExtractOp(op.lhs, self.extract_op.indices)
        extract_rhs_op = TensorExtractOp(op.rhs, self.extract_op.indices)
        call_op = binary_function(extract_lhs_op.result, extract_rhs_op.result)
        rewriter.replace_op(self.extract_op, [extract_lhs_op, extract_rhs_op, call_op])
        rewriter.erase_op(op)


class RewritePadOpPattern(TensorRewritePattern):
    def getOriginalIndex(self, pad_low:int, pad_inner:int, cur_idx:SSAValue) -> tuple[list[Operation], SSAValue, SSAValue]:
        """
        cur_idx = pad_low + ori_idx * (pad_inner + 1)
        Given the current index, returns if it has an original index and its value
        """
        const_0 = getBVConstant(0)
        const_pad_low = getBVConstant(pad_low)
        const_pad_inner_plus_1 = getBVConstant(pad_inner+1)

        cur_idx_uge_pad_low = smt_bv.UgeOp(cur_idx, const_pad_low.res)

        cur_id_minus_pad_low = smt_bv.SubOp(cur_idx,const_pad_low.res)
        urem_op = smt_bv.URemOp(cur_id_minus_pad_low.res, const_pad_inner_plus_1.res)
        urem_op_eq_0 = EqOp(urem_op.res, const_0.res)

        and_op = AndOp(cur_idx_uge_pad_low.res, urem_op_eq_0.res)
        original_idx = smt_bv.UDivOp(cur_id_minus_pad_low.res, const_pad_inner_plus_1.res)
        return ([cur_idx_uge_pad_low, cur_id_minus_pad_low, urem_op, urem_op_eq_0, and_op, original_idx],
                and_op.result,
                original_idx.res)


    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: TensorPadOp, rewriter: PatternRewriter
    ):
        indices = self.extract_op.indices
        pad_low = toTupleInt(op.edge_padding_low)
        pad_inner = toTupleInt(op.interior_padding)
        assert len(indices) == len(pad_inner) == len(pad_low)
        new_ops:list[Operation]=[]
        original_indices_check:list[SSAValue] = []
        original_indices:list[SSAValue] = []

        for i in range(len(pad_low)):
            ops,idx_check,idx = self.getOriginalIndex(pad_low[i], pad_inner[i], indices[i])
            new_ops+=ops
            original_indices_check.append(idx_check)
            original_indices.append(idx)

        and_op = AndOp(*original_indices_check)
        original_extract_op = TensorExtractOp(op.operand, original_indices)
        ite_op = IteOp(and_op.result, original_extract_op.result, op.padding_value)
        rewriter.replace_op(self.extract_op, new_ops+[and_op, original_extract_op, ite_op])
        rewriter.erase_op(op)


class RewriteSliceOpPattern(TensorRewritePattern):
    def getOriginalIndex(self, start_index:int, stride:int, cur_idx:SSAValue) -> tuple[list[Operation], SSAValue]:
        """
        cur_idx = pad_low + ori_idx * (pad_inner + 1)
        Given the current index, returns if it has an original index and its value
        """
        const_start_index = getBVConstant(start_index)
        const_stride = getBVConstant(stride)

        mul_stride = smt_bv.MulOp(const_stride.res, cur_idx)
        add_start_index = smt_bv.AddOp(const_start_index.res, mul_stride.res)
        return [mul_stride, add_start_index], add_start_index.res


    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: TensorSliceOp, rewriter: PatternRewriter
    ):
        indices = self.extract_op.indices
        start_indices = toTupleInt(op.start_indices)
        strides = toTupleInt(op.strides)
        new_ops:list[Operation]=[]
        original_indices:list[SSAValue] = []

        for i in range(len(strides)):
            ops, idx = self.getOriginalIndex(start_indices[i], strides[i], indices[i])
            new_ops+=ops
            original_indices.append(idx)

        extract_op = TensorExtractOp(op.operand, original_indices)
        rewriter.replace_op(self.extract_op, new_ops+[extract_op])
        rewriter.erase_op(op)



class TestOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        print(op)



class TensorExtractOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorExtractOp, rewriter: PatternRewriter):
        source = op.tensor
        source_parent_op = source.owner
        if isinstance(source_parent_op, ElementwiseUnaryOperation):
            ...
        elif isinstance(source_parent_op, ElementwiseBinaryOperation):
            RewriteElementwiseBinaryOpPattern(op).match_and_rewrite(source_parent_op, rewriter)
        elif isinstance(source_parent_op, TensorTransposeOp):
            RewriteTransposeOpPattern(op).match_and_rewrite(source_parent_op, rewriter)
        elif isinstance(source_parent_op, TensorPadOp):
            RewritePadOpPattern(op).match_and_rewrite(source_parent_op, rewriter)
        elif isinstance(source_parent_op, TensorSliceOp):
            RewriteSliceOpPattern(op).match_and_rewrite(source_parent_op, rewriter)



def insertFunctionBeforeModule(op: ModuleOp):
    block = op.body.block
    first_op = block.first_op
    assert first_op is not None
    while len(elementwise_binary_function_set) > 0:
        function_op = elementwise_binary_function_set.pop()
        block.insert_op_before(function_op, first_op)

    while len(elementwise_unary_function_set) > 0:
        function_op = elementwise_unary_function_set.pop()
        block.insert_op_before(function_op, first_op)



class RewriteSMTTensor(ModulePass):
    name = "rewrite-smt-tensor"

    def apply(self, ctx: Context, op: ModuleOp):
        initElementwiseIntFunction()

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    TensorExtractOpPattern()
                ]
            ), walk_reverse=True
        )
        walker.rewrite_module(op)

        insertFunctionBeforeModule(op)