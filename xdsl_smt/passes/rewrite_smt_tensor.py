from abc import ABC
from typing import Callable

from xdsl.dialects.smt import AndOp

from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv

from xdsl_smt.dialects.smt_dialect import EqOp, DeclareFunOp, IteOp
from xdsl.ir import Operation, SSAValue
from xdsl_smt.dialects.smt_tensor_dialect import (
    ElementwiseBinaryOperation,
    TensorTransposeOp,
    ElementwiseUnaryOperation,
    TensorPadOp,
    INDEX_WIDTH,
    toTupleInt,
    TensorSliceOp,
    TensorBroadcastInDimOp,
    TensorConcatenateOp,
    TensorIotaOp,
)
from xdsl.dialects.builtin import FunctionType, ModuleOp
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


bv_constants: dict[int, smt_bv.ConstantOp] = {}


def getBVConstant(x: int) -> smt_bv.ConstantOp:
    global bv_constants
    if x not in bv_constants:
        bv_constants[x] = smt_bv.ConstantOp.from_int_value(x, INDEX_WIDTH)
    return bv_constants[x]


class TensorRewritePattern(RewritePattern, ABC):
    extract_op: TensorExtractOp

    def __init__(self, extract_op):
        self.extract_op = extract_op
        super().__init__()


class RewriteTransposeOpPattern(TensorRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorTransposeOp, rewriter: PatternRewriter):
        extract_op = self.extract_op
        permutations = op.permutation.get_values()
        new_indices: list[SSAValue] = []
        for i in permutations:
            new_indices.append(extract_op.indices[i])
        new_extract_op = TensorExtractOp(op.operand, new_indices)
        rewriter.replace_op(extract_op, new_extract_op)
        rewriter.erase_matched_op()


class RewriteBroadcastInDimPattern(TensorRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorBroadcastInDimOp, rewriter: PatternRewriter):
        extract_op = self.extract_op
        broadcast_dimensions = op.get_broadcast_dimensions()
        new_indices: list[SSAValue] = []
        const_0 = getBVConstant(0)
        operand_type = op.operand.type
        assert isinstance(operand_type, SMTTensorType)
        operand_shape = operand_type.get_shape()
        assert len(operand_shape) == len(broadcast_dimensions)
        for i, dim in zip(broadcast_dimensions, operand_shape):
            if dim == 0:
                new_indices.append(const_0.res)
            else:
                new_indices.append(extract_op.indices[i])
        new_extract_op = TensorExtractOp(op.operand, new_indices)
        rewriter.replace_op(extract_op, new_extract_op)
        rewriter.erase_matched_op()


def toFuncName(name: str) -> str:
    return name.replace(".", "_")


elementwise_unary_function_set: set[DeclareFunOp] = set()
elementwise_unary_functions: dict[str, Callable[[SSAValue], list[Operation]]] = {}


elementwise_binary_function_set: set[DeclareFunOp] = set()
elementwise_binary_functions: dict[
    str, Callable[[SSAValue, SSAValue], list[Operation]]
] = {}


def initElementwiseIntFunction():
    global elementwise_binary_functions
    global elementwise_unary_functions
    elementwise_binary_functions["smt_tensor_add"] = lambda x, y: [smt_bv.AddOp(x, y)]
    elementwise_binary_functions["smt_tensor_subtract"] = lambda x, y: [
        smt_bv.SubOp(x, y)
    ]
    elementwise_binary_functions["smt_tensor_multiply"] = lambda x, y: [
        smt_bv.MulOp(x, y)
    ]

    def get_maximum_ops(lhs: SSAValue, rhs: SSAValue) -> list[Operation]:
        less_than_op = smt_bv.SltOp(lhs, rhs)
        ite_op = IteOp(less_than_op.res, rhs, lhs)
        return [less_than_op, ite_op]

    elementwise_binary_functions["smt_tensor_maximum"] = get_maximum_ops

    def get_minimum_ops(lhs: SSAValue, rhs: SSAValue) -> list[Operation]:
        less_than_op = smt_bv.SltOp(lhs, rhs)
        ite_op = IteOp(less_than_op.res, lhs, rhs)
        return [less_than_op, ite_op]

    elementwise_binary_functions["smt_tensor_minimum"] = get_minimum_ops

    def get_abs_ops(val: SSAValue) -> list[Operation]:
        neg_op = smt_bv.NegOp(val)
        less_than_op = smt_bv.SltOp(val, neg_op.res)
        ite_op = IteOp(less_than_op.res, neg_op.res, val)
        return [neg_op, less_than_op, ite_op]

    elementwise_unary_functions["smt_tensor_abs"] = get_abs_ops
    elementwise_unary_functions["smt_tensor_negate"] = lambda x: [smt_bv.NegOp(x)]


def getElementwiseBinaryFunction(op_name: str, element_type: Attribute):
    global elementwise_binary_function_set
    global elementwise_binary_functions
    if op_name not in elementwise_binary_functions:
        element_uf_type = FunctionType.from_lists(
            [element_type, element_type], [element_type]
        )
        defun_op = DeclareFunOp(element_uf_type, op_name)
        elementwise_binary_function_set.add(defun_op)
        elementwise_binary_functions[op_name] = lambda x, y: [
            CallOp(defun_op.ret, [x, y])
        ]
    return elementwise_binary_functions[op_name]


def getElementwiseUnaryFunction(op_name: str, element_type: Attribute):
    global elementwise_unary_function_set
    global elementwise_unary_functions
    if op_name not in elementwise_unary_functions:
        element_uf_type = FunctionType.from_lists([element_type], [element_type])
        defun_op = DeclareFunOp(element_uf_type, op_name)
        elementwise_unary_function_set.add(defun_op)
        elementwise_unary_functions[op_name] = lambda x: [CallOp(defun_op.ret, [x])]
    return elementwise_unary_functions[op_name]


class RewriteIotaOpPattern(TensorRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorIotaOp, rewriter: PatternRewriter):
        element_type = self.extract_op.result.type
        assert isinstance(element_type, smt_bv.BitVectorType)
        iota_dimension = op.get_iota_dimension()
        result_width = element_type.width.data
        result_ops = []
        if result_width > INDEX_WIDTH:
            result_ops = [
                smt_bv.ZeroExtendOp(
                    self.extract_op.indices[iota_dimension], element_type
                )
            ]
        elif result_width < INDEX_WIDTH:
            result_ops = [
                smt_bv.ExtractOp(
                    self.extract_op.indices[iota_dimension], result_width, 0
                )
            ]
        if result_ops != []:
            rewriter.replace_op(self.extract_op, result_ops)
        else:
            rewriter.replace_all_uses_with(
                self.extract_op.result, self.extract_op.indices[iota_dimension]
            )
            rewriter.erase_op(self.extract_op)
        rewriter.erase_matched_op()


class RewriteElementwiseUnaryOpPattern(TensorRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ElementwiseUnaryOperation, rewriter: PatternRewriter
    ):
        element_type = self.extract_op.result.type
        op_name = toFuncName(op.name)
        unary_function = getElementwiseUnaryFunction(op_name, element_type)
        extract_op_op = TensorExtractOp(op.op, self.extract_op.indices)
        call_ops = unary_function(extract_op_op.result)
        rewriter.replace_op(self.extract_op, [extract_op_op] + call_ops)
        rewriter.erase_matched_op()


class RewriteElementwiseBinaryOpPattern(TensorRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ElementwiseBinaryOperation, rewriter: PatternRewriter
    ):
        element_type = self.extract_op.result.type
        op_name = toFuncName(op.name)
        binary_function = getElementwiseBinaryFunction(op_name, element_type)
        extract_lhs_op = TensorExtractOp(op.lhs, self.extract_op.indices)
        extract_rhs_op = TensorExtractOp(op.rhs, self.extract_op.indices)
        call_ops = binary_function(extract_lhs_op.result, extract_rhs_op.result)
        rewriter.replace_op(
            self.extract_op, [extract_lhs_op, extract_rhs_op] + call_ops
        )
        rewriter.erase_op(op)


class RewriteConcatenateOpPattern(TensorRewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorConcatenateOp, rewriter: PatternRewriter):
        dim = op.get_dimension()
        inputs_dim: list[int] = []
        operands = [operand for operand in op.inputs]
        for operand in operands:
            tensor_type = operand.type
            assert isinstance(tensor_type, SMTTensorType)
            inputs_dim.append(tensor_type.get_shape()[dim])
        for i in range(1, len(inputs_dim)):
            inputs_dim[i] += inputs_dim[i - 1]

        ite_ops: list[Operation] = []
        last_index_val = self.extract_op.indices[dim]
        last_operand_val = operands[0]

        extract_indices = [val for val in self.extract_op.indices]
        for i, operand in zip(inputs_dim[:-1], operands[1:]):
            bv_const = getBVConstant(i)
            ge_shape_op = smt_bv.SgeOp(self.extract_op.indices[dim], bv_const.res)
            new_index_op = smt_bv.SubOp(self.extract_op.indices[dim], bv_const.res)
            ite_index_op = IteOp(ge_shape_op.res, new_index_op.res, last_index_val)
            ite_operand_op = IteOp(ge_shape_op.res, operand, last_operand_val)
            ite_ops += [ge_shape_op, new_index_op, ite_index_op, ite_operand_op]
            last_index_val = ite_index_op.res
            last_operand_val = ite_operand_op.res

        extract_indices[dim] = last_index_val
        extract_op = TensorExtractOp(last_operand_val, extract_indices)

        rewriter.replace_op(self.extract_op, ite_ops + [extract_op])
        rewriter.erase_op(op)


class RewritePadOpPattern(TensorRewritePattern):
    def getOriginalIndex(
        self, pad_low: int, pad_inner: int, cur_idx: SSAValue
    ) -> tuple[list[Operation], SSAValue, SSAValue]:
        """
        cur_idx = pad_low + ori_idx * (pad_inner + 1)
        Given the current index, returns if it has an original index and its value
        """
        const_0 = getBVConstant(0)
        const_pad_low = getBVConstant(pad_low)
        const_pad_inner_plus_1 = getBVConstant(pad_inner + 1)

        cur_idx_uge_pad_low = smt_bv.UgeOp(cur_idx, const_pad_low.res)

        cur_id_minus_pad_low = smt_bv.SubOp(cur_idx, const_pad_low.res)
        urem_op = smt_bv.URemOp(cur_id_minus_pad_low.res, const_pad_inner_plus_1.res)
        urem_op_eq_0 = EqOp(urem_op.res, const_0.res)

        and_op = AndOp(cur_idx_uge_pad_low.res, urem_op_eq_0.res)
        original_idx = smt_bv.UDivOp(
            cur_id_minus_pad_low.res, const_pad_inner_plus_1.res
        )
        return (
            [
                cur_idx_uge_pad_low,
                cur_id_minus_pad_low,
                urem_op,
                urem_op_eq_0,
                and_op,
                original_idx,
            ],
            and_op.result,
            original_idx.res,
        )

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorPadOp, rewriter: PatternRewriter):
        indices = self.extract_op.indices
        pad_low = toTupleInt(op.edge_padding_low)
        pad_inner = toTupleInt(op.interior_padding)
        assert len(indices) == len(pad_inner) == len(pad_low)
        new_ops: list[Operation] = []
        original_indices_check: list[SSAValue] = []
        original_indices: list[SSAValue] = []

        for i in range(len(pad_low)):
            ops, idx_check, idx = self.getOriginalIndex(
                pad_low[i], pad_inner[i], indices[i]
            )
            new_ops += ops
            original_indices_check.append(idx_check)
            original_indices.append(idx)

        and_op = AndOp(*original_indices_check)
        original_extract_op = TensorExtractOp(op.operand, original_indices)
        ite_op = IteOp(and_op.result, original_extract_op.result, op.padding_value)
        rewriter.replace_op(
            self.extract_op, new_ops + [and_op, original_extract_op, ite_op]
        )
        rewriter.erase_op(op)


class RewriteSliceOpPattern(TensorRewritePattern):
    def getOriginalIndex(
        self, start_index: int, stride: int, cur_idx: SSAValue
    ) -> tuple[list[Operation], SSAValue]:
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
    def match_and_rewrite(self, op: TensorSliceOp, rewriter: PatternRewriter):
        indices = self.extract_op.indices
        start_indices = toTupleInt(op.start_indices)
        strides = toTupleInt(op.strides)
        new_ops: list[Operation] = []
        original_indices: list[SSAValue] = []

        for i in range(len(strides)):
            ops, idx = self.getOriginalIndex(start_indices[i], strides[i], indices[i])
            new_ops += ops
            original_indices.append(idx)

        extract_op = TensorExtractOp(op.operand, original_indices)
        rewriter.replace_op(self.extract_op, new_ops + [extract_op])
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
            RewriteElementwiseUnaryOpPattern(op).match_and_rewrite(
                source_parent_op, rewriter
            )
        elif isinstance(source_parent_op, ElementwiseBinaryOperation):
            RewriteElementwiseBinaryOpPattern(op).match_and_rewrite(
                source_parent_op, rewriter
            )
        elif isinstance(source_parent_op, TensorTransposeOp):
            RewriteTransposeOpPattern(op).match_and_rewrite(source_parent_op, rewriter)
        elif isinstance(source_parent_op, TensorPadOp):
            RewritePadOpPattern(op).match_and_rewrite(source_parent_op, rewriter)
        elif isinstance(source_parent_op, TensorSliceOp):
            RewriteSliceOpPattern(op).match_and_rewrite(source_parent_op, rewriter)
        elif isinstance(source_parent_op, TensorIotaOp):
            RewriteIotaOpPattern(op).match_and_rewrite(source_parent_op, rewriter)
        elif isinstance(source_parent_op, TensorBroadcastInDimOp):
            RewriteBroadcastInDimPattern(op).match_and_rewrite(
                source_parent_op, rewriter
            )
        elif isinstance(source_parent_op, TensorConcatenateOp):
            RewriteConcatenateOpPattern(op).match_and_rewrite(
                source_parent_op, rewriter
            )


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
        initElementwiseIntFunction()

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([TensorExtractOpPattern()]), walk_reverse=True
        )
        walker.rewrite_module(op)

        insertFunctionBeforeModule(op)
        insertConstantsBeforeModule(op)
