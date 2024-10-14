from typing import Mapping, Sequence
from dataclasses import dataclass
from xdsl.utils.hints import isa
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.irdl import IRDLOperation
from xdsl.dialects.builtin import AnyIntegerAttr, IntegerAttr, IntegerType
import xdsl.dialects.comb as comb

from xdsl_smt.dialects import hw_dialect as hw
from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_utils_dialect as smt_utils
from xdsl_smt.passes.lower_to_smt import SMTLowerer
from xdsl_smt.semantics.builtin_semantics import IntegerAttrSemantics
from xdsl_smt.semantics.semantics import OperationSemantics
from xdsl_smt.semantics.arith_semantics import SimplePurePoisonSemantics


def cast_integer_type(
    value: SSAValue, target_type: smt_bv.BitVectorType, rewriter: PatternRewriter
) -> SSAValue:
    assert isa(value.type, smt_bv.BitVectorType)
    if value.type.width == target_type.width:
        return value

    if value.type.width.data > target_type.width.data:
        extract_op = smt_bv.ExtractOp(value, target_type.width.data - 1, 0)
        rewriter.insert_op_before_matched_op([extract_op])
        return extract_op.res

    zero = smt_bv.ConstantOp(0, target_type.width.data - value.type.width.data)
    concat_op = smt_bv.ConcatOp(zero.res, value)
    rewriter.insert_op_before_matched_op([zero, concat_op])
    return concat_op.res


class ConstantSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        value_value = attributes["value"]
        if isinstance(value_value, Attribute):
            assert isa(value_value, AnyIntegerAttr)
            value_value = IntegerAttrSemantics().get_semantics(value_value, rewriter)
        poison_op = smt.ConstantBoolOp(False)
        rewriter.insert_op_before_matched_op(poison_op)
        res_op = smt_utils.PairOp(value_value, poison_op.res)
        rewriter.insert_op_before_matched_op(res_op)
        return ((res_op.res,), effect_state)


@dataclass
class VariadicSemantics(SimplePurePoisonSemantics):
    comb_op_type: type[IRDLOperation]
    smt_op_type: type[IRDLOperation]
    empty_value: int

    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        res_type = SMTLowerer.lower_type(results[0])
        assert isa(res_type, smt_utils.PairType[smt_bv.BitVectorType, smt.BoolType])
        if len(operands) == 0:
            constant = smt_bv.ConstantOp(self.empty_value, res_type.first.width)
            rewriter.insert_op_before_matched_op(constant)
            return ((constant.res, None),)

        current_val = operands[0]

        for operand in operands[1:]:
            new_op = self.smt_op_type.create(
                operands=[current_val, operand],
                result_types=[res_type.first],
            )
            current_val = new_op.results[0]
            rewriter.insert_op_before_matched_op(new_op)

        return ((current_val, None),)


@dataclass
class TrivialBinOpSemantics(SimplePurePoisonSemantics):
    comb_op_type: type[Operation]
    smt_op_type: type[Operation]

    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(result := results[0], IntegerType)
        new_op = self.smt_op_type.create(
            operands=operands,
            result_types=[smt_bv.BitVectorType(result.width)],
        )
        rewriter.insert_op_before_matched_op([new_op])
        return ((new_op.results[0], None),)


class ICmpSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        predicate_value = attributes["predicate"]
        if isinstance(predicate_value, Attribute):
            assert isa(predicate_value, AnyIntegerAttr)
            predicate_value = IntegerAttrSemantics().get_semantics(
                predicate_value, rewriter
            )

        zero_i1 = smt_bv.ConstantOp(0, 1)
        one_i1 = smt_bv.ConstantOp(1, 1)
        rewriter.insert_op_before_matched_op([zero_i1, one_i1])

        # Predicate 0: eq
        value_0 = smt.EqOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op([value_0])

        # Predicate 1: ne
        value_1 = smt.DistinctOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op([value_1])

        # Predicate 2: slt
        value_2 = smt_bv.SltOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op([value_2])

        # Predicate 3: sle
        value_3 = smt_bv.SleOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op([value_3])

        # Predicate 4: sgt
        value_4 = smt_bv.SgtOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op([value_4])

        # Predicate 5: sge
        value_5 = smt_bv.SgeOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op([value_5])

        # Predicate 6: ult
        value_6 = smt_bv.UltOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op([value_6])

        # Predicate 7: ule
        value_7 = smt_bv.UleOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op([value_7])

        # Predicate 8: ugt
        value_8 = smt_bv.UgtOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op([value_8])

        # Predicate 9: uge
        value_9 = smt_bv.UgeOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op([value_9])

        zero = smt_bv.ConstantOp(0, 64)
        one = smt_bv.ConstantOp(1, 64)
        two = smt_bv.ConstantOp(2, 64)
        three = smt_bv.ConstantOp(3, 64)
        four = smt_bv.ConstantOp(4, 64)
        five = smt_bv.ConstantOp(5, 64)
        six = smt_bv.ConstantOp(6, 64)
        seven = smt_bv.ConstantOp(7, 64)
        eight = smt_bv.ConstantOp(8, 64)
        rewriter.insert_op_before_matched_op(
            [zero, one, two, three, four, five, six, seven, eight]
        )

        eq_0 = smt.EqOp(predicate_value, zero.res)
        eq_1 = smt.EqOp(predicate_value, one.res)
        eq_2 = smt.EqOp(predicate_value, two.res)
        eq_3 = smt.EqOp(predicate_value, three.res)
        eq_4 = smt.EqOp(predicate_value, four.res)
        eq_5 = smt.EqOp(predicate_value, five.res)
        eq_6 = smt.EqOp(predicate_value, six.res)
        eq_7 = smt.EqOp(predicate_value, seven.res)
        eq_8 = smt.EqOp(predicate_value, eight.res)
        rewriter.insert_op_before_matched_op(
            [eq_0, eq_1, eq_2, eq_3, eq_4, eq_5, eq_6, eq_7, eq_8]
        )

        # Switch case on predicate
        ite_8 = smt.IteOp(eq_8.res, value_8.res, value_9.res)
        ite_7 = smt.IteOp(eq_7.res, value_7.res, ite_8.res)
        ite_6 = smt.IteOp(eq_6.res, value_6.res, ite_7.res)
        ite_5 = smt.IteOp(eq_5.res, value_5.res, ite_6.res)
        ite_4 = smt.IteOp(eq_4.res, value_4.res, ite_5.res)
        ite_3 = smt.IteOp(eq_3.res, value_3.res, ite_4.res)
        ite_2 = smt.IteOp(eq_2.res, value_2.res, ite_3.res)
        ite_1 = smt.IteOp(eq_1.res, value_1.res, ite_2.res)
        ite_0 = smt.IteOp(eq_0.res, value_0.res, ite_1.res)
        rewriter.insert_op_before_matched_op(
            [ite_8, ite_7, ite_6, ite_5, ite_4, ite_3, ite_2, ite_1, ite_0]
        )
        to_int = smt.IteOp(ite_0.res, one_i1.res, zero_i1.res)
        rewriter.insert_op_before_matched_op(to_int)
        return ((to_int.res, None),)


class ParitySemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        assert operands[0].type.width.data > 0
        bits: list[SSAValue] = []

        for i in range(operands[0].type.width.data):
            extract = smt_bv.ExtractOp(operands[0], i, i)
            bits.append(extract.res)
            rewriter.insert_op_before_matched_op(extract)

        res = bits[0]
        for bit in bits[1:]:
            xor_op = smt_bv.XorOp(res, bit)
            rewriter.insert_op_before_matched_op(xor_op)
            res = xor_op.res

        return ((res, None),)


class ExtractSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        low_bit = attributes["low_bit"]
        assert isinstance(results[0], IntegerType)
        if isinstance(low_bit, Attribute):
            if not isa(low_bit, IntegerAttr[IntegerType]):
                raise Exception(
                    "comb.extract expects an IntegrAttr constant or an SSA value"
                )
            low_bit_op = smt_bv.ConstantOp(low_bit)
            rewriter.insert_op_before_matched_op(low_bit_op)
            low_bit = low_bit_op.res

        assert isa(operands[0].type, smt_bv.BitVectorType)
        low_bit = cast_integer_type(low_bit, operands[0].type, rewriter)

        shift_op = smt_bv.LShrOp(operands[0], low_bit)
        rewriter.insert_op_before_matched_op(shift_op)
        extract_op = smt_bv.ExtractOp(shift_op.res, results[0].width.data - 1, 0)
        rewriter.insert_op_before_matched_op(extract_op)
        return ((extract_op.res, None),)


class ReplicateSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isa(operands[0].type, smt_bv.BitVectorType)
        assert isinstance(results[0], IntegerType)
        num_repetition = results[0].width.data // operands[0].type.width.data
        current_val = operands[0]

        for _ in range(num_repetition - 1):
            new_op = smt_bv.ConcatOp(current_val, operands[0])
            current_val = new_op.results[0]
            rewriter.insert_op_before_matched_op(new_op)

        return ((current_val, None),)


class MuxSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        one = smt_bv.ConstantOp(1, 1)
        eq = smt.EqOp(operands[0], one.res)
        ite = smt.IteOp(eq.res, operands[1], operands[2])
        rewriter.insert_op_before_matched_op([one, eq, ite])
        return ((ite.res, None),)


comb_semantics: dict[type[Operation], OperationSemantics] = {
    hw.ConstantOp: ConstantSemantics(),
    comb.AddOp: VariadicSemantics(comb.AddOp, smt_bv.AddOp, 0),
    comb.MulOp: VariadicSemantics(comb.MulOp, smt_bv.MulOp, 1),
    comb.DivUOp: TrivialBinOpSemantics(comb.DivUOp, smt_bv.UDivOp),
    comb.DivSOp: TrivialBinOpSemantics(comb.DivSOp, smt_bv.SDivOp),
    comb.ModUOp: TrivialBinOpSemantics(comb.ModUOp, smt_bv.URemOp),
    comb.ModSOp: TrivialBinOpSemantics(comb.ModSOp, smt_bv.SRemOp),
    comb.ShlOp: TrivialBinOpSemantics(comb.ShlOp, smt_bv.ShlOp),
    comb.ShrUOp: TrivialBinOpSemantics(comb.ShrUOp, smt_bv.LShrOp),
    comb.ShrSOp: TrivialBinOpSemantics(comb.ShrSOp, smt_bv.AShrOp),
    comb.SubOp: TrivialBinOpSemantics(comb.SubOp, smt_bv.SubOp),
    comb.OrOp: VariadicSemantics(comb.OrOp, smt_bv.OrOp, 0),
    comb.AndOp: VariadicSemantics(comb.AndOp, smt_bv.AndOp, 1),
    comb.XorOp: VariadicSemantics(comb.XorOp, smt_bv.XorOp, 0),
    comb.ICmpOp: ICmpSemantics(),
    comb.ParityOp: ParitySemantics(),
    comb.ExtractOp: ExtractSemantics(),
    comb.ConcatOp: VariadicSemantics(comb.ConcatOp, smt_bv.ConcatOp, 0),
    comb.ReplicateOp: ReplicateSemantics(),
    comb.MuxOp: MuxSemantics(),
}
