from typing import cast, Callable
from dataclasses import dataclass
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.ir import Operation

from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_array_dialect as smt_array
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
    TensorSubtractOp, TensorAbsOp,
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
from ..dialects.smt_array_dialect import SelectOp
from ..dialects.smt_bitvector_dialect import BinaryBVOp, UnaryBVOp

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


def lower_tensor_type(type: Attribute) -> Attribute:
    if isinstance(type, SMTTensorType):
        result = BoolType()
        bv_type = smt_bv.BitVectorType(32)
        tensor_type = smt_bv.BitVectorType(32)
        for _ in type.shape:
            result = PairType(bv_type, result)
            tensor_type = smt_array.ArrayType(bv_type, tensor_type)
        return PairType(tensor_type, result)
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
                    block, i + 1, lower_tensor_type(typ)
                )
                arg.replace_by(dimension_arg)
                rewriter.erase_block_argument(arg)
            else:
                i += 1
        old_typ = op.ret.type
        assert isinstance(old_typ, FunctionType)
        new_inputs = [arg.type for arg in block.args]
        new_outputs = [lower_tensor_type(ty) for ty in old_typ.outputs]
        op.ret.type = FunctionType.from_attrs(
            ArrayAttr[Attribute](new_inputs), ArrayAttr[Attribute](new_outputs)
        )


def get_tensor_and_index(val: SSAValue) -> tuple[list[Operation], SSAValue, SSAValue]:
    if not isinstance(val.type, PairType):
        raise ValueError("Can't get tensor value")
    result: list[Operation] = [FirstOp(val), SecondOp(val)]
    cur_tensor = result[0].results[0]
    cur_index = result[1].results[0]
    return result, cur_tensor, cur_index


def get_tensor_value(
    cur_tensor: SSAValue, cur_index: SSAValue
) -> tuple[list[Operation], SSAValue]:
    result: list[Operation] = []
    while isinstance(cur_index.type, PairType):
        first_op = FirstOp(cur_index)
        second_op = SecondOp(cur_index)
        select_op = SelectOp(cur_tensor, first_op.res)
        result += [first_op, second_op, select_op]
        cur_tensor = select_op.res
        cur_index = second_op.res
    assert not isinstance(cur_tensor.type, smt_array.ArrayType)
    return result, cur_tensor


class LowerElementwiseBinaryOpPattern(RewritePattern):
    def get_binary_op(self, op: ElementwiseBinaryOperation) -> type[BinaryBVOp]:
        if isinstance(op, TensorAddOp):
            return smt_bv.AddOp
        elif isinstance(op, TensorSubtractOp):
            return smt_bv.SubOp
        raise ValueError("Don't support binary op for" + str(type(op)))

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ElementwiseBinaryOperation, rewriter: PatternRewriter
    ):
        if not isa((pair_type := op.result.type), AnyPairType):
            new_tensor_decl_op = DeclareConstOp(op.lhs.type)
            new_tensor_op, new_tensor, new_tensor_index = get_tensor_and_index(
                new_tensor_decl_op.res
            )
            lhs_tensor_op, lhs_tensor, lhs_tensor_index = get_tensor_and_index(op.lhs)
            rhs_tensor_op, rhs_tensor, rhs_tensor_index = get_tensor_and_index(op.rhs)
            eq_ops: list[EqOp] = [
                EqOp(lhs_tensor_index, rhs_tensor_index),
                EqOp(new_tensor_index, lhs_tensor_index),
            ]
            new_tensor_value_op, new_tensor_val = get_tensor_value(
                new_tensor, new_tensor_index
            )
            lhs_tensor_value_op, lhs_tensor_val = get_tensor_value(
                lhs_tensor, lhs_tensor_index
            )
            rhs_tensor_value_op, rhs_tensor_val = get_tensor_value(
                rhs_tensor, rhs_tensor_index
            )
            binary_op = self.get_binary_op(op)(lhs_tensor_val, rhs_tensor_val)
            eq_ops.append(EqOp(new_tensor_val, binary_op.res))
            assert_ops = [AssertOp(eq.res) for eq in eq_ops]

            rewriter.insert_op_before_matched_op(
                [new_tensor_decl_op]
                + new_tensor_op
                + lhs_tensor_op
                + rhs_tensor_op
                + new_tensor_value_op
                + lhs_tensor_value_op
                + rhs_tensor_value_op
                + [binary_op]
                + eq_ops
                + assert_ops
            )
            op.result.replace_by(new_tensor_decl_op.res)
            rewriter.erase_matched_op()


class LowerElementwiseUnaryOpPattern(RewritePattern):
    def get_unary_op(self, op: ElementwiseUnaryOperation) -> Callable[[SSAValue], list[Operation]]:
        if isinstance(op, TensorAbsOp):
            def get_abs_ops(val:SSAValue) -> list[Operation]:
                neg_op =smt_bv.NegOp(val)
                less_than_op = smt_bv.SltOp(val, neg_op.res)
                ite_op = smt.IteOp(less_than_op.res, neg_op.res, val)
                return [neg_op, less_than_op, ite_op]
            return get_abs_ops
        raise ValueError("Don't support unary op for" + str(type(op)))

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ElementwiseUnaryOperation, rewriter: PatternRewriter
    ):
        if not isa((pair_type := op.result.type), AnyPairType):
            new_tensor_decl_op = DeclareConstOp(op.lhs.type)
            new_tensor_op, new_tensor, new_tensor_index = get_tensor_and_index(
                new_tensor_decl_op.res
            )
            lhs_tensor_op, lhs_tensor, lhs_tensor_index = get_tensor_and_index(op.lhs)
            rhs_tensor_op, rhs_tensor, rhs_tensor_index = get_tensor_and_index(op.rhs)
            eq_ops: list[EqOp] = [
                EqOp(lhs_tensor_index, rhs_tensor_index),
                EqOp(new_tensor_index, lhs_tensor_index),
            ]
            new_tensor_value_op, new_tensor_val = get_tensor_value(
                new_tensor, new_tensor_index
            )
            lhs_tensor_value_op, lhs_tensor_val = get_tensor_value(
                lhs_tensor, lhs_tensor_index
            )
            rhs_tensor_value_op, rhs_tensor_val = get_tensor_value(
                rhs_tensor, rhs_tensor_index
            )
            binary_op = self.get_binary_op(op)(lhs_tensor_val, rhs_tensor_val)
            eq_ops.append(EqOp(new_tensor_val, binary_op.res))
            assert_ops = [AssertOp(eq.res) for eq in eq_ops]

            rewriter.insert_op_before_matched_op(
                [new_tensor_decl_op]
                + new_tensor_op
                + lhs_tensor_op
                + rhs_tensor_op
                + new_tensor_value_op
                + lhs_tensor_value_op
                + rhs_tensor_value_op
                + [binary_op]
                + eq_ops
                + assert_ops
            )
            op.result.replace_by(new_tensor_decl_op.res)
            rewriter.erase_matched_op()



class LowerTransposeOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TensorTransposeOp, rewriter: PatternRewriter):
        if not isa((pair_type := op.result.type), AnyPairType):
            cur_tensor_op, cur_tensor, cur_index = get_tensor_and_index(op.operand)

            ops: list[Operation] = []
            index_list: list[SSAValue] = []
            for _ in op.permutation.get_values():
                ops.append(FirstOp(cur_index))
                index_list.append(ops[-1].res)
                ops.append(SecondOp(cur_index))
                cur_index = ops[-1].res

            new_index_list: list[SSAValue] = []
            for i in op.permutation.get_values():
                new_index_list.append(index_list[i])
            for val in new_index_list[::-1]:
                ops.append(PairOp(val, cur_index))
                cur_index = ops[-1].results[0]
            ops.append(PairOp(cur_tensor, cur_index))
            rewriter.insert_op_before_matched_op(cur_tensor_op + ops)
            op.result.replace_by(ops[-1].results[0])
            rewriter.erase_matched_op()


class TestOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        print(op)


class LowerTensor(ModulePass):
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
            )
        )
        walker.rewrite_module(op)

        # Apply DCE pass
        DeadCodeElimination().apply(ctx, op)
