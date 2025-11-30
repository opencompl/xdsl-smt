from abc import ABC
from typing import Callable


from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv

from xdsl_smt.dialects.smt_dialect import DeclareFunOp, IteOp
from xdsl.ir import Operation, SSAValue
from xdsl_smt.dialects.smt_tensor_dialect import (
    ElementwiseBinaryOperation,
    TensorTransposeOp,
    ElementwiseUnaryOperation,
    INDEX_WIDTH,
    TensorExtractOp,
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

from xdsl_smt.dialects.smt_dialect import (
    CallOp,
)


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
