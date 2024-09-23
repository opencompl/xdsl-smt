from dataclasses import dataclass
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.ir import Operation

from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import transfer
from xdsl_smt.passes.lower_to_smt.lower_to_smt import SMTLowerer
from xdsl_smt.dialects.smt_utils_dialect import (
    AnyPairType,
    PairType,
    SecondOp,
    FirstOp,
    PairOp,
)
from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.semantics.semantics import EffectStates, OperationSemantics
from xdsl.ir import Operation, SSAValue, Attribute
from typing import Mapping, Sequence
from xdsl.utils.hints import isa
from xdsl.parser import AnyIntegerAttr
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl_smt.utils.transfer_to_smt_util import (
    get_low_bits,
    set_high_bits,
    set_low_bits,
    count_lzeros,
    count_rzeros,
    count_lones,
    count_rones,
)


def abstract_value_type_lowerer(
    type: Attribute,
) -> PairType[Attribute, Attribute] | None:
    """Lower all types in an abstract value to SMT types
    But the last element is useless, this makes GetOp easier"""
    if isinstance(type, transfer.AbstractValueType):
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
    return None


def transfer_integer_type_lowerer(
    type: Attribute, width: int
) -> smt_bv.BitVectorType | None:
    if isinstance(type, transfer.TransIntegerType):
        return smt_bv.BitVectorType(width)
    return None


class ConstantOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width
        const_value = attributes["value"]
        if isinstance(const_value, SSAValue):
            return ((const_value,), effect_states)

        assert isa(const_value, AnyIntegerAttr)
        const_value = const_value.value.data
        bv_const = smt_bv.ConstantOp(const_value, width)
        rewriter.insert_op_before_matched_op(bv_const)
        return ((bv_const.res,), effect_states)


class GetAllOnesOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width
        const_value = (1 << width.data) - 1

        bv_const = smt_bv.ConstantOp(const_value, width)
        rewriter.insert_op_before_matched_op(bv_const)
        return ((bv_const.res,), effect_states)


class GetBitWidthOpSemantics(OperationSemantics):
    """1. There is no direct API to obtain the bit width of a vector.
    2. All bit width are fixed values.
    As a result, we replace this method with a constant."""

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width
        const_value = width
        bv_const = smt_bv.ConstantOp(const_value, width)
        rewriter.insert_op_before_matched_op(bv_const)
        return ((bv_const.res,), effect_states)


class MakeOpSemantics(OperationSemantics):
    # The last element is useless, getOp won't access it
    # So it can be any bool value
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        false_constant = smt.ConstantBoolOp.from_bool(False)
        argList = operands
        opList: list[Operation] = [PairOp(argList[-1], false_constant.res)]
        result = opList[-1].results[0]
        for ty in reversed(argList[:-1]):
            opList.append(PairOp(ty, result))
            result = opList[-1].results[0]

        rewriter.insert_op_before_matched_op(false_constant)
        for newOp in opList[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.insert_op_before_matched_op(opList[-1])
        return ((opList[-1].results[0],), effect_states)


class GetOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
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

        return ((insertOps[-1].results[0],), effect_states)


@dataclass
class TrivialOpSemantics(OperationSemantics):
    comb_op_type: type[Operation]
    smt_op_type: type[Operation]

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        new_op = self.smt_op_type.create(
            operands=operands,
            result_types=[SMTLowerer.lower_type(results[0])],
        )
        rewriter.insert_op_before_matched_op([new_op])
        return ((new_op.results[0],), effect_states)


class UMulOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        suml_nooverflow = smt_bv.UmulNoOverflowOp.get(operands[0], operands[1])

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(suml_nooverflow.res, b0.res, b1.res)
        poison_op = smt.ConstantBoolOp.from_bool(False)
        res = PairOp(bool_to_bv.res, poison_op.res)
        rewriter.insert_op_before_matched_op(
            [suml_nooverflow, b0, b1, bool_to_bv, poison_op, res]
        )
        return ((res.res,), effect_states)


class ShlOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        assert isinstance(lhs_type := operands[0].type, smt_bv.BitVectorType)
        width = lhs_type.width
        const_value = width
        bv_width = smt_bv.ConstantOp(const_value, width)

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        cmp_op = smt_bv.UgeOp(operands[1], bv_width.res)
        bool_to_bv = smt.IteOp(cmp_op.res, b1.res, b0.res)
        poison_op = smt.ConstantBoolOp.from_bool(False)
        res = PairOp(bool_to_bv.res, poison_op.res)
        rewriter.insert_op_before_matched_op(
            [bv_width, b0, b1, cmp_op, bool_to_bv, poison_op, res]
        )
        return ((res.res,), effect_states)


class IsPowerOf2OpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
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
        poison_op = smt.ConstantBoolOp.from_bool(False)
        res = PairOp(bool_to_bv.res, poison_op.res)
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
        return ((res.res,), effect_states)


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
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        predicate = attributes["predicate"]
        assert isinstance(predicate, IntegerAttr)
        predicate = predicate.value.data

        rewrite_type = self.new_ops[predicate]
        new_op = rewrite_type.create(operands=operands, result_types=[BoolType()])

        resList: list[Operation] = [new_op]

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(resList[-1].results[0], b1.results[0], b0.results[0])

        poison_op = smt.ConstantBoolOp.from_bool(False)
        res_op = PairOp(bool_to_bv.results[0], poison_op.res)

        resList += [b1, b0, bool_to_bv, poison_op, res_op]
        rewriter.insert_op_before_matched_op(resList)
        return ((res_op.res,), effect_states)


class IntersectsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        and_res = smt_bv.AndOp(operands[0], operands[1])
        assert isinstance(and_res.res.type, smt_bv.BitVectorType)
        const_0 = smt_bv.ConstantOp(0, and_res.res.type.width)
        eq_0 = smt.EqOp(and_res.res, const_0.res)

        resList: list[Operation] = [and_res, const_0, eq_0]

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(resList[-1].results[0], b0.results[0], b1.results[0])

        poison_op = smt.ConstantBoolOp.from_bool(False)
        res_op = PairOp(bool_to_bv.results[0], poison_op.res)

        resList += [b1, b0, bool_to_bv, poison_op, res_op]
        rewriter.insert_op_before_matched_op(resList)
        return ((res_op.res,), effect_states)


class CountLOneOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        operand = operands[0]
        resList = count_lones(operand)
        rewriter.insert_op_before_matched_op(resList)
        return ((resList[-1].results[0],), effect_states)


class CountLZeroOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        operand = operands[0]
        resList = count_lzeros(operand)
        rewriter.insert_op_before_matched_op(resList)
        return ((resList[-1].results[0],), effect_states)


class CountROneOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        operand = operands[0]
        resList, afterList = count_rones(operand)
        rewriter.insert_op_before_matched_op(resList)
        rewriter.insert_op_before_matched_op(afterList)
        return ((resList[-1].results[0],), effect_states)


class CountRZeroOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        operand = operands[0]
        resList, afterList = count_rzeros(operand)
        rewriter.insert_op_before_matched_op(resList)
        rewriter.insert_op_before_matched_op(afterList)
        return ((resList[-1].results[0],), effect_states)


class SetHighBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        result = set_high_bits(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(result)
        return ((result[-1].results[0],), effect_states)


class SetLowBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        result = set_low_bits(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(result)
        return ((result[-1].results[0],), effect_states)


class GetLowBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        result = get_low_bits(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(result)
        return ((result[-1].results[0],), effect_states)


class SMinOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        smin_if = smt_bv.SleOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return ((ite_op.res,), effect_states)


class UMinOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        smin_if = smt_bv.UleOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return ((ite_op.res,), effect_states)


class SMaxOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        smin_if = smt_bv.SgtOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return ((ite_op.res,), effect_states)


class UMaxOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        smin_if = smt_bv.UgtOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return ((ite_op.res,), effect_states)


class SelectOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        bv1 = smt_bv.ConstantOp.from_int_value(1, 1)
        bv_val = FirstOp(operands[0])
        eq1 = smt.EqOp(bv_val.res, bv1.res)
        ite_op = smt.IteOp(eq1.res, operands[1], operands[2])
        rewriter.insert_op_before_matched_op([bv1, bv_val, eq1, ite_op])
        return ((ite_op.res,), effect_states)


class RepeatOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        cntOp = operands[1].owner
        print(cntOp)
        print(cntOp.parent_op())
        assert (
            isinstance(cntOp, smt_bv.ConstantOp) and "repeat count has to be a constant"
        )
        cnt = cntOp.value.value.data
        repeatOp = smt_bv.RepeatOp(operands[0], cnt)
        rewriter.insert_op_before_matched_op(repeatOp)
        return ((repeatOp.res,), effect_states)


class ConcatOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        concatOp = smt_bv.ConcatOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(concatOp)
        return ((concatOp.res,), effect_states)


class ExtractOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
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
        extractOp = smt_bv.ExtractOp(operands[0], numBits + bitPosition, bitPosition)
        rewriter.insert_op_before_matched_op(extractOp)
        return ((extractOp.res,), effect_states)


class AddPoisonOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        op_ty = operands[0].type
        assert isinstance(op_ty, smt_bv.BitVectorType)
        bool_false = smt.ConstantBoolOp(False)
        res = PairOp(operands[0], bool_false.res)

        rewriter.insert_op_before_matched_op([bool_false, res])
        return ((res.res,), effect_states)


class RemovePoisonOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
        op_ty = operands[0].type
        assert isinstance(op_ty, PairType)
        res = FirstOp(operands[0])
        rewriter.insert_op_before_matched_op([res])
        return ((res.res,), effect_states)


transfer_semantics: dict[type[Operation], OperationSemantics] = {
    transfer.Constant: ConstantOpSemantics(),
    transfer.AddOp: TrivialOpSemantics(transfer.AddOp, smt_bv.AddOp),
    transfer.MulOp: TrivialOpSemantics(transfer.MulOp, smt_bv.MulOp),
    transfer.OrOp: TrivialOpSemantics(transfer.OrOp, smt_bv.OrOp),
    transfer.AndOp: TrivialOpSemantics(transfer.AndOp, smt_bv.AndOp),
    transfer.XorOp: TrivialOpSemantics(transfer.XorOp, smt_bv.XorOp),
    transfer.SubOp: TrivialOpSemantics(transfer.SubOp, smt_bv.SubOp),
    transfer.NegOp: TrivialOpSemantics(transfer.NegOp, smt_bv.NotOp),
    transfer.ShlOp: TrivialOpSemantics(transfer.ShlOp, smt_bv.ShlOp),
    transfer.ConcatOp: ConcatOpSemantics(),
    transfer.RepeatOp: RepeatOpSemantics(),
    transfer.ExtractOp: ExtractOpSemantics(),
    transfer.UMulOverflowOp: UMulOverflowOpSemantics(),
    transfer.ShlOverflowOp: ShlOverflowOpSemantics(),
    transfer.CmpOp: CmpOpSemantics(),
    transfer.GetOp: GetOpSemantics(),
    transfer.MakeOp: MakeOpSemantics(),
    transfer.GetBitWidthOp: GetBitWidthOpSemantics(),
    transfer.CountLOneOp: CountLOneOpSemantics(),
    transfer.CountLZeroOp: CountLZeroOpSemantics(),
    transfer.CountROneOp: CountROneOpSemantics(),
    transfer.CountRZeroOp: CountRZeroOpSemantics(),
    transfer.SMaxOp: SMaxOpSemantics(),
    transfer.SMinOp: SMinOpSemantics(),
    transfer.UMaxOp: UMaxOpSemantics(),
    transfer.UMinOp: UMinOpSemantics(),
    transfer.SetHighBitsOp: SetHighBitsOpSemantics(),
    transfer.SetLowBitsOp: SetLowBitsOpSemantics(),
    transfer.GetLowBitsOp: GetLowBitsOpSemantics(),
    transfer.SelectOp: SelectOpSemantics(),
    transfer.IsPowerOf2Op: IsPowerOf2OpSemantics(),
    transfer.GetAllOnesOp: GetAllOnesOpSemantics(),
    transfer.IntersectsOp: IntersectsOpSemantics(),
    transfer.AddPoisonOp: AddPoisonOpSemantics(),
    transfer.RemovePoisonOp: RemovePoisonOpSemantics(),
}
