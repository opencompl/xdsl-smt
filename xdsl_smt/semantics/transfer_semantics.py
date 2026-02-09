from dataclasses import dataclass
from typing import Mapping, Sequence

from xdsl.pattern_rewriter import PatternRewriter
from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import transfer
from xdsl_smt.passes.lower_to_smt.smt_lowerer import SMTLowerer
from xdsl_smt.dialects.smt_utils_dialect import (
    AnyPairType,
    PairType,
    SecondOp,
    FirstOp,
    PairOp,
)
from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.semantics.semantics import OperationSemantics, TypeSemantics
from xdsl.ir import Operation, SSAValue, Attribute
from xdsl.utils.hints import isa
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl_smt.utils.transfer_to_smt_util import (
    get_low_bits,
    count_lzeros,
    count_rzeros,
    count_lones,
    count_rones,
    reverse_bits,
    is_non_negative,
    is_negative,
    get_high_bits,
)


class AbstractValueTypeSemantics(TypeSemantics):
    """Lower all types in an abstract value to SMT types
    But the last element is useless, this makes GetOp easier"""

    def lower_type(self, ty: Attribute) -> Attribute:
        """
        If the input type is already a smt type, skip lowering
        """
        if ty.name.startswith("smt"):
            return ty
        return SMTLowerer.lower_type(ty)

    def get_semantics(self, type: Attribute) -> Attribute:
        assert isinstance(type, transfer.AbstractValueType) or isinstance(
            type, transfer.TupleType
        )
        curTy = type.get_fields()[-1]
        isIntegerTy = isinstance(curTy, IntegerType)
        curLoweredTy = SMTLowerer.lower_type(curTy)
        if isIntegerTy:
            assert isa(curLoweredTy, PairType[smt_bv.BitVectorType, BoolType])
            curLoweredTy = curLoweredTy.first
        result: AnyPairType = PairType(curLoweredTy, BoolType())
        for ty in reversed(type.get_fields()[:-1]):
            isIntegerTy = isinstance(ty, IntegerType)
            curLoweredTy = SMTLowerer.lower_type(ty)
            if isIntegerTy:
                assert isa(curLoweredTy, PairType[smt_bv.BitVectorType, BoolType])
                curLoweredTy = curLoweredTy.first
            result: AnyPairType = PairType(curLoweredTy, result)
        return result


@dataclass
class TransferIntegerTypeSemantics(TypeSemantics):
    """Lower an integer type to a bitvector integer."""

    width: int

    def get_semantics(self, type: Attribute) -> Attribute:
        assert isinstance(type, transfer.TransIntegerType)
        return smt_bv.BitVectorType(self.width)


class ConstantOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width
        const_value = attributes["value"]
        if isinstance(const_value, SSAValue):
            return ((const_value,), effect_state)

        assert isa(const_value, IntegerAttr)
        const_value = const_value.value.data
        bv_const = smt_bv.ConstantOp(const_value, width)
        rewriter.insert_op_before_matched_op(bv_const)
        return ((bv_const.res,), effect_state)


class GetAllOnesOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width
        const_value = (1 << width.data) - 1

        bv_const = smt_bv.ConstantOp(const_value, width)
        rewriter.insert_op_before_matched_op(bv_const)
        return ((bv_const.res,), effect_state)


class GetSignedMaxValueOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width
        const_value = (1 << (width.data - 1)) - 1

        bv_const = smt_bv.ConstantOp(const_value, width)
        rewriter.insert_op_before_matched_op(bv_const)
        return ((bv_const.res,), effect_state)


class GetSignedMinValueOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width
        const_value = 1 << (width.data - 1)

        bv_const = smt_bv.ConstantOp(const_value, width)
        rewriter.insert_op_before_matched_op(bv_const)
        return ((bv_const.res,), effect_state)


class GetBitWidthOpSemantics(OperationSemantics):
    """1. There is no direct API to obtain the bit width of a vector.
    2. All bit width are fixed values.
    As a result, we replace this method with a constant."""

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width
        const_value = width
        bv_const = smt_bv.ConstantOp(const_value, width)
        rewriter.insert_op_before_matched_op(bv_const)
        return ((bv_const.res,), effect_state)


class MakeOpSemantics(OperationSemantics):
    # The last element is useless, getOp won't access it
    # So it can be any bool value
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        false_constant = smt.ConstantBoolOp(False)
        argList = operands
        opList: list[Operation] = [PairOp(argList[-1], false_constant.result)]
        result = opList[-1].results[0]
        for ty in reversed(argList[:-1]):
            opList.append(PairOp(ty, result))
            result = opList[-1].results[0]

        rewriter.insert_op_before_matched_op(false_constant)
        for newOp in opList[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.insert_op_before_matched_op(opList[-1])
        return ((opList[-1].results[0],), effect_state)


class GetOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        index = attributes["index"]
        arg = operands[0]
        insertOps: list[Operation] = []
        assert isinstance(index, IntegerAttr)
        index = index.value.data
        while index != 0:
            insertOps.append(SecondOp(arg))
            arg = insertOps[-1].results[0]
            index -= 1
        insertOps.append(FirstOp(arg))
        rewriter.insert_op_before_matched_op(insertOps)

        return ((insertOps[-1].results[0],), effect_state)


@dataclass
class TrivialOpSemantics(OperationSemantics):
    comb_op_type: type[Operation]
    smt_op_type: type[Operation]

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        new_op = self.smt_op_type.create(
            operands=operands,
            result_types=[SMTLowerer.lower_type(results[0])],
        )
        rewriter.insert_op_before_matched_op([new_op])
        return ((new_op.results[0],), effect_state)


def smt_bool_to_bv1(bool_val: SSAValue) -> tuple[SSAValue, list[Operation]]:
    """
    Given a SMT bool variable, this functions return 1 or 0 on bv1
    """
    b1 = smt_bv.ConstantOp.from_int_value(1, 1)
    b0 = smt_bv.ConstantOp.from_int_value(0, 1)
    ite_op = smt.IteOp(bool_val, b1.res, b0.res)
    return ite_op.res, [b1, b0, ite_op]


class UMulOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        umul_overflow = smt_bv.UmulOverflowOp(operands[0], operands[1])
        bv_res, ops = smt_bool_to_bv1(umul_overflow.res)

        poison_op = smt.ConstantBoolOp(False)

        res = PairOp(bv_res, poison_op.result)
        rewriter.insert_op_before_matched_op([umul_overflow] + ops + [poison_op, res])
        return ((res.res,), effect_state)


class SMulOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        smul_overflow = smt_bv.SmulOverflowOp(operands[0], operands[1])
        bv_res, ops = smt_bool_to_bv1(smul_overflow.res)

        poison_op = smt.ConstantBoolOp(False)
        res = PairOp(bv_res, poison_op.result)
        rewriter.insert_op_before_matched_op([smul_overflow] + ops + [poison_op, res])
        return ((res.res,), effect_state)


class UAddOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        uadd_overflow = smt_bv.UaddOverflowOp(operands[0], operands[1])
        bv_res, ops = smt_bool_to_bv1(uadd_overflow.res)

        poison_op = smt.ConstantBoolOp(False)
        res = PairOp(bv_res, poison_op.result)
        rewriter.insert_op_before_matched_op([uadd_overflow] + ops + [poison_op, res])
        return ((res.res,), effect_state)


class SAddOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        sadd_overflow = smt_bv.SaddOverflowOp(operands[0], operands[1])
        bv_res, ops = smt_bool_to_bv1(sadd_overflow.res)

        poison_op = smt.ConstantBoolOp(False)
        res = PairOp(bv_res, poison_op.result)
        rewriter.insert_op_before_matched_op([sadd_overflow] + ops + [poison_op, res])
        return ((res.res,), effect_state)


class USubOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        usub_overflow = smt_bv.UsubOverflowOp(operands[0], operands[1])
        bv_res, ops = smt_bool_to_bv1(usub_overflow.res)

        poison_op = smt.ConstantBoolOp(False)
        res = PairOp(bv_res, poison_op.result)
        rewriter.insert_op_before_matched_op([usub_overflow] + ops + [poison_op, res])
        return ((res.res,), effect_state)


class SSubOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        ssub_overflow = smt_bv.SsubOverflowOp(operands[0], operands[1])
        bv_res, ops = smt_bool_to_bv1(ssub_overflow.res)

        poison_op = smt.ConstantBoolOp(False)
        res = PairOp(bv_res, poison_op.result)
        rewriter.insert_op_before_matched_op([ssub_overflow] + ops + [poison_op, res])
        return ((res.res,), effect_state)


class UShlOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        """
        Overflow = ShAmt >= getBitWidth();
        if (Overflow)
          return APInt(BitWidth, 0);
        Overflow = ShAmt > countl_zero();

        overflow should be ShAmt >= getBitWidth() ||  ShAmt > countl_zero()
        Can't be replaced by ShAmt > countl_zero() because countl_zero() <=getBitWidth()
        """
        assert isinstance(lhs_type := operands[0].type, smt_bv.BitVectorType)
        width = lhs_type.width
        const_value = width
        bv_width = smt_bv.ConstantOp(const_value, width)

        operand = operands[0]
        shift_amount = operands[1]

        shift_amount_ge_bitwidth = smt_bv.UgeOp(shift_amount, bv_width.res)
        countl_zero_ops = count_lzeros(operand)
        lzero_operand = countl_zero_ops[-1].results[0]
        shift_amount_gt_lzero = smt_bv.UgtOp(shift_amount, lzero_operand)
        or_op = smt.OrOp(shift_amount_ge_bitwidth.res, shift_amount_gt_lzero.res)

        overflow_ops = (
            [bv_width, shift_amount_ge_bitwidth]
            + countl_zero_ops
            + [shift_amount_gt_lzero, or_op]
        )

        bv_res, bool_to_bv1_ops = smt_bool_to_bv1(or_op.result)

        poison_op = smt.ConstantBoolOp(False)
        res = PairOp(bv_res, poison_op.result)

        rewriter.insert_op_before_matched_op(
            overflow_ops + bool_to_bv1_ops + [poison_op, res]
        )
        return ((res.res,), effect_state)


class SShlOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        """
        Overflow = ShAmt >= getBitWidth();
        if (Overflow)
            return APInt(BitWidth, 0);
        if (isNonNegative()) // Don't allow sign change.
            Overflow = ShAmt >= countl_zero();
        else
            Overflow = ShAmt >= countl_one();

        overflow should be ShAmt >= getBitWidth() ||
            (isNonNegative()&&ShAmt >= countl_zero() || isNegative()&&ShAmt >= countl_one())
        """
        assert isinstance(lhs_type := operands[0].type, smt_bv.BitVectorType)
        width = lhs_type.width
        const_value = width
        bv_width = smt_bv.ConstantOp(const_value, width)

        operand = operands[0]
        shift_amount = operands[1]

        # ShAmt >= getBitWidth()
        shift_amount_ge_bitwidth = smt_bv.UgeOp(shift_amount, bv_width.res)

        # isNonNegative()
        is_non_negative_ops = is_non_negative(operand)
        is_non_negative_operand = is_non_negative_ops[-1].results[0]

        # ShAmt >= countl_zero()
        countl_zero_ops = count_lzeros(operand)
        lzero_operand = countl_zero_ops[-1].results[0]
        shift_amount_ge_lzero = smt_bv.UgeOp(shift_amount, lzero_operand)

        # isNegative()
        is_negative_ops = is_negative(operand)
        is_negative_operand = is_negative_ops[-1].results[0]

        # ShAmt >= countl_one()
        countl_one_ops = count_lones(operand)
        lone_operand = countl_one_ops[-1].results[0]
        shift_amount_ge_lone = smt_bv.UgeOp(shift_amount, lone_operand)

        # isNonNegative()&&ShAmt >= countl_zero()
        and_op = smt.AndOp(is_non_negative_operand, shift_amount_ge_lzero.res)

        # isNegative()&&ShAmt >= countl_one()
        and1_op = smt.AndOp(is_negative_operand, shift_amount_ge_lone.res)

        # isNonNegative()&&ShAmt >= countl_zero() || isNegative()&&ShAmt >= countl_one()
        or_op = smt.OrOp(and_op.result, and1_op.result)

        final_or_op = smt.OrOp(shift_amount_ge_bitwidth.res, or_op.result)

        overflow_ops = (
            [bv_width, shift_amount_ge_bitwidth]
            + is_non_negative_ops
            + countl_zero_ops
            + [shift_amount_ge_lzero]
            + is_negative_ops
            + countl_one_ops
            + [shift_amount_ge_lone]
            + [and_op, and1_op, or_op, final_or_op]
        )

        bv_res, bool_to_bv1_ops = smt_bool_to_bv1(final_or_op.result)

        poison_op = smt.ConstantBoolOp(False)
        res = PairOp(bv_res, poison_op.result)

        rewriter.insert_op_before_matched_op(
            overflow_ops + bool_to_bv1_ops + [poison_op, res]
        )
        return ((res.res,), effect_state)


class IsPowerOf2OpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert isinstance(lhs_type := operands[0].type, smt_bv.BitVectorType)
        width = lhs_type.width
        const_value = width.data

        b1 = smt_bv.ConstantOp.from_int_value(1, const_value)
        b0 = smt_bv.ConstantOp.from_int_value(0, const_value)
        b1_1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0_1 = smt_bv.ConstantOp.from_int_value(0, 1)
        op_minus_one = smt_bv.SubOp(operands[0], b1.res)
        and_op = smt_bv.AndOp(operands[0], op_minus_one.res)
        eq_op = smt.EqOp(b0.res, and_op.res)
        bool_to_bv = smt.IteOp(eq_op.res, b1_1.res, b0_1.res)
        poison_op = smt.ConstantBoolOp(False)
        res = PairOp(bool_to_bv.res, poison_op.result)
        rewriter.insert_op_before_matched_op(
            [
                b0,
                b1,
                b1_1,
                b0_1,
                op_minus_one,
                and_op,
                eq_op,
                bool_to_bv,
                poison_op,
                res,
            ]
        )
        return ((res.res,), effect_state)


class CmpOpSemantics(OperationSemantics):
    new_ops = [
        smt.EqOp,
        smt.DistinctOp,
        smt_bv.SltOp,
        smt_bv.SleOp,
        smt_bv.SgtOp,
        smt_bv.SgeOp,
        smt_bv.UltOp,
        smt_bv.UleOp,
        smt_bv.UgtOp,
        smt_bv.UgeOp,
    ]

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        predicate = attributes["predicate"]
        assert isinstance(predicate, IntegerAttr)
        predicate = predicate.value.data

        rewrite_type = self.new_ops[predicate]
        new_op = rewrite_type.create(operands=operands, result_types=[BoolType()])

        resList: list[Operation] = [new_op]

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(resList[-1].results[0], b1.results[0], b0.results[0])

        poison_op = smt.ConstantBoolOp(False)
        res_op = PairOp(bool_to_bv.results[0], poison_op.result)

        resList += [b1, b0, bool_to_bv, poison_op, res_op]
        rewriter.insert_op_before_matched_op(resList)
        return ((res_op.res,), effect_state)


class IntersectsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        and_res = smt_bv.AndOp(operands[0], operands[1])
        assert isinstance(and_res.res.type, smt_bv.BitVectorType)
        const_0 = smt_bv.ConstantOp(0, and_res.res.type.width)
        eq_0 = smt.EqOp(and_res.res, const_0.res)

        resList: list[Operation] = [and_res, const_0, eq_0]

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(resList[-1].results[0], b0.results[0], b1.results[0])

        poison_op = smt.ConstantBoolOp(False)
        res_op = PairOp(bool_to_bv.results[0], poison_op.result)

        resList += [b1, b0, bool_to_bv, poison_op, res_op]
        rewriter.insert_op_before_matched_op(resList)
        return ((res_op.res,), effect_state)


class CountLOneOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        operand = operands[0]
        resList = count_lones(operand)
        rewriter.insert_op_before_matched_op(resList)
        return ((resList[-1].results[0],), effect_state)


class CountLZeroOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        operand = operands[0]
        resList = count_lzeros(operand)
        rewriter.insert_op_before_matched_op(resList)
        return ((resList[-1].results[0],), effect_state)


class CountROneOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        operand = operands[0]
        resList, afterList = count_rones(operand)
        rewriter.insert_op_before_matched_op(resList)
        rewriter.insert_op_before_matched_op(afterList)
        return ((resList[-1].results[0],), effect_state)


class CountRZeroOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        operand = operands[0]
        resList, afterList = count_rzeros(operand)
        rewriter.insert_op_before_matched_op(resList)
        rewriter.insert_op_before_matched_op(afterList)
        return ((resList[-1].results[0],), effect_state)


class PopCountOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        operand = operands[0]
        assert isinstance(bv_type := operand.type, smt_bv.BitVectorType)
        width = bv_type.width

        const_zero = smt_bv.ConstantOp(0, width)
        ops: list[Operation] = [const_zero]
        acc = const_zero.res

        for i in range(width.data):
            extract = smt_bv.ExtractOp(operand, i, i)
            ops.append(extract)
            zext = smt_bv.ZeroExtendOp(extract.res, bv_type)
            ops.append(zext)
            add = smt_bv.AddOp(acc, zext.res)
            ops.append(add)
            acc = add.res

        rewriter.insert_op_before_matched_op(ops)
        return ((acc,), effect_state)


class SetHighBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        arg = operands[0]
        count = operands[1]
        assert isinstance(bv_type := arg.type, smt_bv.BitVectorType)

        const_bw = smt_bv.ConstantOp(bv_type.width, bv_type.width)
        const_one = smt_bv.ConstantOp(1, bv_type.width)

        umin = smt_bv.UltOp(count, const_bw.res)
        clamped_count = smt.IteOp(umin.res, count, const_bw.res)

        sub = smt_bv.SubOp(const_bw.res, clamped_count.res)
        shl = smt_bv.ShlOp(const_one.res, clamped_count.res)
        sub2 = smt_bv.SubOp(shl.res, const_one.res)
        shl2 = smt_bv.ShlOp(sub2.res, sub.res)
        or_op = smt_bv.OrOp(arg, shl2.res)

        rewriter.insert_op_before_matched_op(
            [const_bw, const_one, umin, clamped_count, sub, shl, sub2, shl2, or_op]
        )

        return ((or_op.res,), effect_state)


class SetLowBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        arg = operands[0]
        count = operands[1]
        assert isinstance(bv_type := arg.type, smt_bv.BitVectorType)

        const_one = smt_bv.ConstantOp(1, bv_type.width)

        shl = smt_bv.ShlOp(const_one.res, count)
        sub = smt_bv.SubOp(shl.res, const_one.res)
        or_op = smt_bv.OrOp(arg, sub.res)

        rewriter.insert_op_before_matched_op([const_one, shl, sub, or_op])

        return ((or_op.res,), effect_state)


class SetSignBitOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        operand = operands[0]
        operand_type = operand.type
        assert isinstance(operand_type, smt_bv.BitVectorType)
        width = operand_type.width.data
        sign_bit = smt_bv.ConstantOp(1 << (width - 1), width)
        or_op = smt_bv.OrOp(sign_bit.res, operand)
        result = [sign_bit, or_op]

        rewriter.insert_op_before_matched_op(result)
        return ((result[-1].results[0],), effect_state)


class ClearSignBitOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        operand = operands[0]
        operand_type = operand.type
        assert isinstance(operand_type, smt_bv.BitVectorType)
        width = operand_type.width.data
        signed_max_value = smt_bv.ConstantOp((1 << (width - 1)) - 1, width)
        and_op = smt_bv.AndOp(signed_max_value.res, operand)
        result = [signed_max_value, and_op]

        rewriter.insert_op_before_matched_op(result)
        return ((result[-1].results[0],), effect_state)


class GetHighBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        result = get_high_bits(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(result)
        return ((result[-1].results[0],), effect_state)


class GetLowBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        result = get_low_bits(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(result)
        return ((result[-1].results[0],), effect_state)


class ClearHighBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        arg = operands[0]
        count = operands[1]
        assert isinstance(bv_type := arg.type, smt_bv.BitVectorType)

        const_bw = smt_bv.ConstantOp(bv_type.width, bv_type.width)
        one = smt_bv.ConstantOp(1, bv_type.width)

        umin = smt_bv.UltOp(count, const_bw.res)
        new_count = smt.IteOp(umin.res, count, const_bw.res)

        # mask = (1 << (width - count)) - 1
        sub = smt_bv.SubOp(const_bw.res, new_count.res)
        shl = smt_bv.ShlOp(one.res, sub.res)
        mask = smt_bv.SubOp(shl.res, one.res)
        masked = smt_bv.AndOp(arg, mask.res)

        rewriter.insert_op_before_matched_op(
            [const_bw, one, umin, new_count, sub, shl, mask, masked]
        )

        return ((masked.res,), effect_state)


class ClearLowBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        arg = operands[0]
        count = operands[1]
        assert isinstance(bv_type := arg.type, smt_bv.BitVectorType)

        const_one = smt_bv.ConstantOp(1, bv_type.width)

        # mask = ~((1 << count) - 1)
        shl = smt_bv.ShlOp(const_one.res, count)
        sub = smt_bv.SubOp(shl.res, const_one.res)
        not_mask = smt_bv.NotOp(sub.res)
        masked = smt_bv.AndOp(arg, not_mask.res)

        rewriter.insert_op_before_matched_op([const_one, shl, sub, not_mask, masked])

        return ((masked.res,), effect_state)


class SMinOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        smin_if = smt_bv.SleOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return ((ite_op.res,), effect_state)


class UMinOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        smin_if = smt_bv.UleOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return ((ite_op.res,), effect_state)


class SMaxOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        smin_if = smt_bv.SgtOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return ((ite_op.res,), effect_state)


class UMaxOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        smin_if = smt_bv.UgtOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return ((ite_op.res,), effect_state)


class SelectOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        bv1 = smt_bv.ConstantOp.from_int_value(1, 1)
        bv_val = FirstOp(operands[0])
        eq1 = smt.EqOp(bv_val.res, bv1.res)
        ite_op = smt.IteOp(eq1.res, operands[1], operands[2])
        rewriter.insert_op_before_matched_op([bv1, bv_val, eq1, ite_op])
        return ((ite_op.res,), effect_state)


class RepeatOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        cntOp = operands[1].owner
        print(cntOp)
        print(cntOp.parent_op())
        assert (
            isinstance(cntOp, smt_bv.ConstantOp) and "repeat count has to be a constant"
        )
        cnt = cntOp.value.value.data
        repeatOp = smt_bv.RepeatOp(operands[0], cnt)
        rewriter.insert_op_before_matched_op(repeatOp)
        return ((repeatOp.res,), effect_state)


class ConcatOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        concatOp = smt_bv.ConcatOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(concatOp)
        return ((concatOp.res,), effect_state)


class ExtractOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        numBitsOp = operands[1].owner
        bitPositionOp = operands[2].owner
        assert (
            isinstance(numBitsOp, smt_bv.ConstantOp) and "num bits has to be a constant"
        )
        assert (
            isinstance(bitPositionOp, smt_bv.ConstantOp)
            and "bit position has to be a constant"
        )
        numBits = numBitsOp.value.value.data
        bitPosition = bitPositionOp.value.value.data
        extractOp = smt_bv.ExtractOp(
            operands[0], numBits + bitPosition - 1, bitPosition
        )
        rewriter.insert_op_before_matched_op(extractOp)
        return ((extractOp.res,), effect_state)


class AddPoisonOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        op_ty = operands[0].type
        assert isinstance(op_ty, smt_bv.BitVectorType)
        bool_false = smt.ConstantBoolOp(False)
        res = PairOp(operands[0], bool_false.result)

        rewriter.insert_op_before_matched_op([bool_false, res])
        return ((res.res,), effect_state)


class RemovePoisonOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        op_ty = operands[0].type
        assert isinstance(op_ty, PairType)
        res = FirstOp(operands[0])
        rewriter.insert_op_before_matched_op([res])
        return ((res.res,), effect_state)


class ReverseBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        op_ty = operands[0].type
        assert isinstance(op_ty, smt_bv.BitVectorType)
        res = reverse_bits(operands[0], rewriter)
        return ((res,), effect_state)


transfer_semantics: dict[type[Operation], OperationSemantics] = {
    transfer.Constant: ConstantOpSemantics(),
    transfer.AddOp: TrivialOpSemantics(transfer.AddOp, smt_bv.AddOp),
    transfer.MulOp: TrivialOpSemantics(transfer.MulOp, smt_bv.MulOp),
    transfer.OrOp: TrivialOpSemantics(transfer.OrOp, smt_bv.OrOp),
    transfer.AndOp: TrivialOpSemantics(transfer.AndOp, smt_bv.AndOp),
    transfer.XorOp: TrivialOpSemantics(transfer.XorOp, smt_bv.XorOp),
    transfer.SubOp: TrivialOpSemantics(transfer.SubOp, smt_bv.SubOp),
    transfer.NegOp: TrivialOpSemantics(transfer.NegOp, smt_bv.NotOp),
    transfer.UDivOp: TrivialOpSemantics(transfer.UDivOp, smt_bv.UDivOp),
    transfer.SDivOp: TrivialOpSemantics(transfer.SDivOp, smt_bv.SDivOp),
    transfer.URemOp: TrivialOpSemantics(transfer.URemOp, smt_bv.URemOp),
    transfer.SRemOp: TrivialOpSemantics(transfer.SRemOp, smt_bv.SRemOp),
    transfer.LShrOp: TrivialOpSemantics(transfer.LShrOp, smt_bv.LShrOp),
    transfer.AShrOp: TrivialOpSemantics(transfer.AShrOp, smt_bv.AShrOp),
    transfer.ShlOp: TrivialOpSemantics(transfer.ShlOp, smt_bv.ShlOp),
    transfer.ConcatOp: ConcatOpSemantics(),
    transfer.RepeatOp: RepeatOpSemantics(),
    transfer.ExtractOp: ExtractOpSemantics(),
    transfer.UMulOverflowOp: UMulOverflowOpSemantics(),
    transfer.SMulOverflowOp: SMulOverflowOpSemantics(),
    transfer.UAddOverflowOp: UAddOverflowOpSemantics(),
    transfer.SAddOverflowOp: SAddOverflowOpSemantics(),
    transfer.USubOverflowOp: USubOverflowOpSemantics(),
    transfer.SSubOverflowOp: SSubOverflowOpSemantics(),
    transfer.UShlOverflowOp: UShlOverflowOpSemantics(),
    transfer.SShlOverflowOp: SShlOverflowOpSemantics(),
    transfer.CmpOp: CmpOpSemantics(),
    transfer.GetOp: GetOpSemantics(),
    transfer.MakeOp: MakeOpSemantics(),
    transfer.GetBitWidthOp: GetBitWidthOpSemantics(),
    transfer.CountLOneOp: CountLOneOpSemantics(),
    transfer.CountLZeroOp: CountLZeroOpSemantics(),
    transfer.CountROneOp: CountROneOpSemantics(),
    transfer.CountRZeroOp: CountRZeroOpSemantics(),
    transfer.PopCountOp: PopCountOpSemantics(),
    transfer.SMaxOp: SMaxOpSemantics(),
    transfer.SMinOp: SMinOpSemantics(),
    transfer.UMaxOp: UMaxOpSemantics(),
    transfer.UMinOp: UMinOpSemantics(),
    transfer.SetHighBitsOp: SetHighBitsOpSemantics(),
    transfer.SetLowBitsOp: SetLowBitsOpSemantics(),
    transfer.SetSignBitOp: SetSignBitOpSemantics(),
    transfer.ClearSignBitOp: ClearSignBitOpSemantics(),
    transfer.GetHighBitsOp: GetHighBitsOpSemantics(),
    transfer.GetLowBitsOp: GetLowBitsOpSemantics(),
    transfer.ClearHighBitsOp: ClearHighBitsOpSemantics(),
    transfer.ClearLowBitsOp: ClearLowBitsOpSemantics(),
    transfer.SelectOp: SelectOpSemantics(),
    transfer.IsPowerOf2Op: IsPowerOf2OpSemantics(),
    transfer.GetAllOnesOp: GetAllOnesOpSemantics(),
    transfer.GetSignedMaxValueOp: GetSignedMaxValueOpSemantics(),
    transfer.GetSignedMinValueOp: GetSignedMinValueOpSemantics(),
    transfer.IntersectsOp: IntersectsOpSemantics(),
    transfer.AddPoisonOp: AddPoisonOpSemantics(),
    transfer.RemovePoisonOp: RemovePoisonOpSemantics(),
    transfer.ReverseBitsOp: ReverseBitsOpSemantics(),
}
