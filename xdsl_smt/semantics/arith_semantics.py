from abc import abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence
from xdsl.ir import Operation, SSAValue, Attribute
from xdsl.parser import AnyIntegerAttr
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.dialects.builtin import IntegerType
from xdsl.utils.hints import isa
from xdsl_smt.semantics.builtin_semantics import IntegerAttrSemantics

from xdsl_smt.semantics.semantics import OperationSemantics

from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_utils_dialect as smt_utils
from xdsl_smt.dialects.effects import ub_effect as smt_ub
import xdsl.dialects.arith as arith


def get_int_value_and_poison(
    val: SSAValue, rewriter: PatternRewriter
) -> tuple[SSAValue, SSAValue]:
    value = smt_utils.FirstOp(val)
    poison = smt_utils.SecondOp(val)
    rewriter.insert_op_before_matched_op([value, poison])
    return value.res, poison.res


def reduce_poison_values(
    operands: Sequence[SSAValue], rewriter: PatternRewriter
) -> tuple[Sequence[SSAValue], SSAValue]:
    if not operands:
        no_poison_op = smt.ConstantBoolOp(False)
        rewriter.insert_op_before_matched_op([no_poison_op])
        return operands, no_poison_op.res

    values = list[SSAValue]()
    value, result_poison = get_int_value_and_poison(operands[0], rewriter)
    values.append(value)

    for operand in operands[1:]:
        value, poison = get_int_value_and_poison(operand, rewriter)
        values.append(value)
        merge_poison = smt.OrOp(result_poison, poison)
        result_poison = merge_poison.res
        rewriter.insert_op_before_matched_op(merge_poison)

    return values, result_poison


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

        no_poison = smt.ConstantBoolOp.from_bool(False)
        res = smt_utils.PairOp(value_value, no_poison.res)
        rewriter.insert_op_before_matched_op([no_poison, res])
        return ((res.res,), effect_state)


@dataclass
class SimplePoisonSemantics(OperationSemantics):
    """
    Semantics of an operation that propagates poison, and sometimes produce it.
    May have an effect.
    """

    @abstractmethod
    def get_simple_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[tuple[SSAValue, SSAValue | None]], SSAValue]:
        pass

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        if effect_state is None:
            raise ValueError("Effect state is required for arith operations")
        operands, propagated_poison = reduce_poison_values(operands, rewriter)
        value_results, new_effect_state = self.get_simple_semantics(
            operands, results, attributes, effect_state, rewriter
        )
        value_with_poison_results: list[SSAValue] = []
        for value, new_poison in value_results:
            if isinstance(new_poison, SSAValue):
                poison = smt.OrOp(new_poison, propagated_poison)
                pair = smt_utils.PairOp(value, poison.res)
                rewriter.insert_op_before_matched_op([poison, pair])
                value_with_poison_results.append(pair.res)
            else:
                pair = smt_utils.PairOp(value, propagated_poison)
                rewriter.insert_op_before_matched_op([pair])
                value_with_poison_results.append(pair.res)

        return (value_with_poison_results, new_effect_state)


@dataclass
class SimplePurePoisonSemantics(SimplePoisonSemantics):
    """
    Semantics of an operation that propagates poison, and sometimes produce it.
    Does not touch any effect.
    """

    @abstractmethod
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        pass

    def get_simple_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[tuple[SSAValue, SSAValue | None]], SSAValue]:
        return (
            self.get_pure_semantics(operands, results, attributes, rewriter),
            effect_state,
        )


def single_binop_semantics(
    op_type: type[smt_bv.BinaryBVOp],
) -> type[SimplePurePoisonSemantics]:
    class SingleBinopSemantics(SimplePurePoisonSemantics):
        def get_pure_semantics(
            self,
            operands: Sequence[SSAValue],
            results: Sequence[Attribute],
            attributes: Mapping[str, Attribute | SSAValue],
            rewriter: PatternRewriter,
        ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
            op = op_type(operands[0], operands[1])
            rewriter.insert_op_before_matched_op([op])
            return ((op.res, None),)

    return SingleBinopSemantics


AddiSemantics = single_binop_semantics(smt_bv.AddOp)
SubiSemantics = single_binop_semantics(smt_bv.SubOp)
MuliSemantics = single_binop_semantics(smt_bv.MulOp)
AndiSemantics = single_binop_semantics(smt_bv.AndOp)
OriSemantics = single_binop_semantics(smt_bv.OrOp)
XoriSemantics = single_binop_semantics(smt_bv.XorOp)


class MulSIExtendedSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        assert isinstance(results[1], IntegerType)
        width = results[0].width.data

        lhs_extend = smt_bv.SignExtendOp(operands[0], smt_bv.BitVectorType(width * 2))
        rhs_extend = smt_bv.SignExtendOp(operands[1], smt_bv.BitVectorType(width * 2))

        res_extend = smt_bv.MulOp(lhs_extend.res, rhs_extend.res)

        low_bits = smt_bv.ExtractOp(res_extend.res, width - 1, 0)
        high_bits = smt_bv.ExtractOp(res_extend.res, 2 * width - 1, width)

        rewriter.insert_op_before_matched_op(
            [lhs_extend, rhs_extend, res_extend, low_bits, high_bits]
        )
        return ((low_bits.res, None), (high_bits.res, None))


class MulUIExtendedSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        assert isinstance(results[1], IntegerType)
        width = results[0].width.data

        lhs_extend = smt_bv.ZeroExtendOp(operands[0], smt_bv.BitVectorType(width * 2))
        rhs_extend = smt_bv.ZeroExtendOp(operands[1], smt_bv.BitVectorType(width * 2))

        res_extend = smt_bv.MulOp(lhs_extend.res, rhs_extend.res)

        low_bits = smt_bv.ExtractOp(res_extend.res, width - 1, 0)
        high_bits = smt_bv.ExtractOp(res_extend.res, 2 * width - 1, width)

        rewriter.insert_op_before_matched_op(
            [lhs_extend, rhs_extend, res_extend, low_bits, high_bits]
        )
        return ((low_bits.res, None), (high_bits.res, None))


class ShliSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        width = results[0].width.data

        value_op = smt_bv.ShlOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op([value_op])

        # If the shift amount is greater than the width of the value, poison
        width_op = smt_bv.ConstantOp(width, width)
        shift_amount_too_big = smt_bv.UgtOp(operands[1], width_op.res)

        rewriter.insert_op_before_matched_op([width_op, shift_amount_too_big])
        return ((value_op.res, shift_amount_too_big.res),)


class DivsiSemantics(SimplePoisonSemantics):
    def get_simple_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[tuple[SSAValue, SSAValue | None]], SSAValue]:
        assert isinstance(results[0], IntegerType)
        width = results[0].width.data

        # Check for division by zero
        zero = smt_bv.ConstantOp(0, width)
        is_div_by_zero = smt.EqOp(operands[1], zero.res)

        # Check for underflow
        minimum_value = smt_bv.ConstantOp(2 ** (width - 1), width)
        minus_one = smt_bv.ConstantOp(2**width - 1, width)
        lhs_is_min_val = smt.EqOp(operands[0], minimum_value.res)
        rhs_is_minus_one = smt.EqOp(operands[1], minus_one.res)
        is_underflow = smt.AndOp(lhs_is_min_val.res, rhs_is_minus_one.res)

        # New poison cases
        trigger_ub = smt_ub.TriggerOp(effect_state)
        new_state = smt.IteOp(is_div_by_zero.res, trigger_ub.res, effect_state)

        # Operation result
        value_op = smt_bv.SDivOp(operands[0], operands[1])

        rewriter.insert_op_before_matched_op(
            [
                zero,
                is_div_by_zero,
                minimum_value,
                minus_one,
                lhs_is_min_val,
                rhs_is_minus_one,
                is_underflow,
                trigger_ub,
                new_state,
                value_op,
            ]
        )
        return (((value_op.res, is_underflow.res),), new_state.res)


class DivuiSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        width = results[0].width.data

        # Check for division by zero
        zero = smt_bv.ConstantOp(0, width)
        is_div_by_zero = smt.EqOp(operands[1], zero.res)

        # Operation result
        value_op = smt_bv.UDivOp(operands[0], operands[1])

        rewriter.insert_op_before_matched_op(
            [
                zero,
                is_div_by_zero,
                value_op,
            ]
        )
        return ((value_op.res, is_div_by_zero.res),)


class RemsiSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        width = results[0].width.data

        # Check for remainder by zero
        zero = smt_bv.ConstantOp(0, width)
        is_rem_by_zero = smt.EqOp(operands[1], zero.res)

        # Operation result
        value_op = smt_bv.SRemOp(operands[0], operands[1])

        rewriter.insert_op_before_matched_op(
            [
                zero,
                is_rem_by_zero,
                value_op,
            ]
        )
        return ((value_op.res, is_rem_by_zero.res),)


class RemuiSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        width = results[0].width.data

        # Check for remainder by zero
        zero = smt_bv.ConstantOp(0, width)
        is_rem_by_zero = smt.EqOp(operands[1], zero.res)

        # Operation result
        value_op = smt_bv.URemOp(operands[0], operands[1])

        rewriter.insert_op_before_matched_op(
            [
                zero,
                is_rem_by_zero,
                value_op,
            ]
        )
        return ((value_op.res, is_rem_by_zero.res),)


class ShrsiSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        width = results[0].width.data

        # Check for shift amount greater than width
        width_op = smt_bv.ConstantOp(width, width)
        shift_amount_too_big = smt_bv.UgtOp(operands[1], width_op.res)

        # Operation result
        value_op = smt_bv.AShrOp(operands[0], operands[1])

        rewriter.insert_op_before_matched_op(
            [
                width_op,
                shift_amount_too_big,
                value_op,
            ]
        )
        return ((value_op.res, shift_amount_too_big.res),)


class ShruiSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        width = results[0].width.data

        # Check for shift amount greater than width
        width_op = smt_bv.ConstantOp(width, width)
        shift_amount_too_big = smt_bv.UgtOp(operands[1], width_op.res)

        # Operation result
        value_op = smt_bv.LShrOp(operands[0], operands[1])

        rewriter.insert_op_before_matched_op(
            [
                width_op,
                shift_amount_too_big,
                value_op,
            ]
        )
        return ((value_op.res, shift_amount_too_big.res),)


class MaxsiSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        # Operation result
        cond_op = smt_bv.SgtOp(operands[0], operands[1])
        value_op = smt.IteOp(cond_op.res, operands[0], operands[1])

        rewriter.insert_op_before_matched_op(
            [
                cond_op,
                value_op,
            ]
        )
        return ((value_op.res, None),)


class MaxuiSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        # Operation result
        cond_op = smt_bv.UgtOp(operands[0], operands[1])
        value_op = smt.IteOp(cond_op.res, operands[0], operands[1])

        rewriter.insert_op_before_matched_op(
            [
                cond_op,
                value_op,
            ]
        )
        return ((value_op.res, None),)


class MinsiSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        # Operation result
        cond_op = smt_bv.SleOp(operands[0], operands[1])
        value_op = smt.IteOp(cond_op.res, operands[0], operands[1])

        rewriter.insert_op_before_matched_op(
            [
                cond_op,
                value_op,
            ]
        )
        return ((value_op.res, None),)


class MinuiSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        # Operation result
        cond_op = smt_bv.UleOp(operands[0], operands[1])
        value_op = smt.IteOp(cond_op.res, operands[0], operands[1])

        rewriter.insert_op_before_matched_op(
            [
                cond_op,
                value_op,
            ]
        )
        return ((value_op.res, None),)


class CmpiSemantics(SimplePurePoisonSemantics):
    predicates = [
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

    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        predicate_opcode = attributes["predicate"]
        if not isinstance(predicate_opcode, SSAValue):
            assert isa(predicate_opcode, AnyIntegerAttr)
            predicate_opcode = IntegerAttrSemantics().get_semantics(
                predicate_opcode, rewriter
            )

        opcode_values: list[SSAValue] = []
        for predicate in self.predicates:
            op = predicate(operands[0], operands[1])
            opcode_values.append(op.res)
            rewriter.insert_op_before_matched_op([op])

        # The choice between the last two opcodes
        before_last_one_opcode = smt_bv.ConstantOp(len(self.predicates) - 2, 64)
        last_two_choice = smt.EqOp(before_last_one_opcode.res, predicate_opcode)
        last_two_opcodes = smt.IteOp(
            last_two_choice.res, opcode_values[-2], opcode_values[-1]
        )
        rewriter.insert_op_before_matched_op(
            [before_last_one_opcode, last_two_choice, last_two_opcodes]
        )

        # Get the actual result depending on the opcode value
        current_value = last_two_opcodes.res
        for i in range(len(self.predicates) - 3, -1, -1):
            current_opcode = smt_bv.ConstantOp(i, 64)
            current_choice = smt.EqOp(current_opcode.res, predicate_opcode)
            current_opcodes = smt.IteOp(
                current_choice.res, opcode_values[i], current_value
            )
            rewriter.insert_op_before_matched_op(
                [current_opcode, current_choice, current_opcodes]
            )
            current_value = current_opcodes.res

        zero = smt_bv.ConstantOp(0, 1)
        one = smt_bv.ConstantOp(1, 1)
        res_value = smt.IteOp(current_value, one.res, zero.res)
        rewriter.insert_op_before_matched_op([zero, one, res_value])

        return ((res_value.res, None),)


class SelectSemantics(OperationSemantics):
    # select poison a, b -> poison
    # select true, a, poison -> a

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        # Get all values and poisons
        cond_val, cond_poi = get_int_value_and_poison(operands[0], rewriter)
        tr_val, tr_poi = get_int_value_and_poison(operands[1], rewriter)
        fls_val, fls_poi = get_int_value_and_poison(operands[2], rewriter)

        # Get the resulting value depending on the condition
        one = smt_bv.ConstantOp(1, 1)
        to_smt_bool = smt.EqOp(cond_val, one.res)
        res_val = smt.IteOp(to_smt_bool.res, tr_val, fls_val)
        br_poi = smt.IteOp(to_smt_bool.res, tr_poi, fls_poi)

        # If the condition is poison, the result is poison
        res_poi = smt.IteOp(cond_poi, cond_poi, br_poi.res)

        res_op = smt_utils.PairOp(res_val.res, res_poi.res)

        rewriter.insert_op_before_matched_op(
            [one, to_smt_bool, res_val, br_poi, res_poi, res_op]
        )
        return ((res_op.res,), effect_state)


class TruncISemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        new_width = results[0].width.data

        res = smt_bv.ExtractOp(operands[0], new_width - 1, 0)
        rewriter.insert_op_before_matched_op([res])
        return ((res.res, None),)


class ExtUISemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        new_width = results[0].width.data

        op = smt_bv.ZeroExtendOp(operands[0], smt_bv.BitVectorType(new_width))
        rewriter.insert_op_before_matched_op([op])
        return ((op.res, None),)


class ExtSISemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        new_width = results[0].width.data

        op = smt_bv.SignExtendOp(operands[0], smt_bv.BitVectorType(new_width))
        rewriter.insert_op_before_matched_op([op])
        return ((op.res, None),)


class CeilDivUISemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        width = results[0].width.data

        zero = smt_bv.ConstantOp(0, width)
        one = smt_bv.ConstantOp(1, width)

        # Check for division by zero
        is_rhs_zero = smt.EqOp(zero.res, operands[1])

        # We need to check if the lhs is zero, so we don't underflow later on
        is_lhs_zero = smt.EqOp(zero.res, operands[0])

        # Compute floor((lhs - 1) / rhs) + 1
        lhs_minus_one = smt_bv.SubOp(operands[0], one.res)
        floor_div = smt_bv.UDivOp(lhs_minus_one.res, operands[1])
        nonzero_res = smt_bv.AddOp(floor_div.res, one.res)

        # If the lhs is zero, the result is zero
        value_res = smt.IteOp(is_lhs_zero.res, zero.res, nonzero_res.res)

        rewriter.insert_op_before_matched_op(
            [
                zero,
                one,
                is_rhs_zero,
                is_lhs_zero,
                lhs_minus_one,
                floor_div,
                nonzero_res,
                value_res,
            ]
        )

        return ((value_res.res, is_rhs_zero.res),)


class CeilDivSISemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        width = results[0].width.data

        # Check for underflow
        minimum_value = smt_bv.ConstantOp(2 ** (width - 1), width)
        minus_one = smt_bv.ConstantOp(2**width - 1, width)
        one = smt_bv.ConstantOp(1, width)
        lhs_is_min_val = smt.EqOp(operands[0], minimum_value.res)
        rhs_is_minus_one = smt.EqOp(operands[1], minus_one.res)
        is_underflow = smt.AndOp(lhs_is_min_val.res, rhs_is_minus_one.res)

        # Check for division by zero
        zero = smt_bv.ConstantOp(0, width)
        is_div_by_zero = smt.EqOp(zero.res, operands[1])

        # Poison if underflow or division by zero or previous poison
        introduce_poison = smt.OrOp(is_underflow.res, is_div_by_zero.res)

        # Division when the result is positive
        # we do ((lhs - 1) / rhs) + 1 for lhs and rhs positive
        # and ((lhs + 1)) / lhs) + 1 for lhs and rhs negative
        is_lhs_positive = smt_bv.SltOp(operands[0], zero.res)
        opposite_one = smt.IteOp(is_lhs_positive.res, minus_one.res, one.res)
        lhs_minus_one = smt_bv.SubOp(operands[0], opposite_one.res)
        floor_div = smt_bv.SDivOp(lhs_minus_one.res, operands[1])
        positive_res = smt_bv.AddOp(floor_div.res, one.res)

        # If the result is non positive
        # We do - ((- lhs) / rhs)
        minus_lhs = smt_bv.SubOp(zero.res, operands[0])
        neg_floor_div = smt_bv.SDivOp(minus_lhs.res, operands[1])
        nonpositive_res = smt_bv.SubOp(zero.res, neg_floor_div.res)

        # Check if the result is positive
        is_rhs_positive = smt_bv.SltOp(operands[1], zero.res)
        is_lhs_negative = smt_bv.SltOp(zero.res, operands[0])
        is_rhs_negative = smt_bv.SltOp(zero.res, operands[1])
        are_both_positive = smt.AndOp(
            is_lhs_positive.res,
            is_rhs_positive.res,
        )
        are_both_negative = smt.AndOp(
            is_lhs_negative.res,
            is_rhs_negative.res,
        )
        is_positive = smt.OrOp(are_both_positive.res, are_both_negative.res)

        value_res = smt.IteOp(is_positive.res, positive_res.res, nonpositive_res.res)

        rewriter.insert_op_before_matched_op(
            [
                minimum_value,
                minus_one,
                one,
                lhs_is_min_val,
                rhs_is_minus_one,
                is_underflow,
                zero,
                is_div_by_zero,
                introduce_poison,
                is_lhs_positive,
                opposite_one,
                lhs_minus_one,
                floor_div,
                positive_res,
                minus_lhs,
                neg_floor_div,
                nonpositive_res,
                is_rhs_positive,
                is_lhs_negative,
                is_rhs_negative,
                are_both_positive,
                are_both_negative,
                is_positive,
                value_res,
            ]
        )
        return ((value_res.res, introduce_poison.res),)


class FloorDivSISemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        assert isinstance(results[0], IntegerType)
        width = results[0].width.data

        # Check for underflow
        minimum_value = smt_bv.ConstantOp(2 ** (width - 1), width)
        minus_one = smt_bv.ConstantOp(2**width - 1, width)
        lhs_is_min_val = smt.EqOp(operands[0], minimum_value.res)
        rhs_is_minus_one = smt.EqOp(operands[1], minus_one.res)
        is_underflow = smt.AndOp(lhs_is_min_val.res, rhs_is_minus_one.res)

        # Check for division by zero
        zero = smt_bv.ConstantOp(0, width)
        is_div_by_zero = smt.EqOp(zero.res, operands[1])

        # Poison if underflow or division by zero or previous poison
        introduce_poison = smt.OrOp(is_underflow.res, is_div_by_zero.res)

        # Compute a division rounded by zero
        value_op = smt_bv.SDivOp(operands[0], operands[1])

        # If the result is negative, subtract 1 if the remainder is not 0
        is_lhs_negative = smt_bv.SltOp(operands[0], zero.res)
        is_rhs_negative = smt_bv.SltOp(operands[1], zero.res)
        is_negative = smt.XorOp(is_lhs_negative.res, is_rhs_negative.res)
        remainder = smt_bv.SRemOp(operands[0], operands[1])
        is_remainder_not_zero = smt.DistinctOp(remainder.res, zero.res)
        subtract_one = smt_bv.AddOp(value_op.res, minus_one.res)
        should_subtract_one = smt.AndOp(is_negative.res, is_remainder_not_zero.res)
        res_value_op = smt.IteOp(
            should_subtract_one.res, subtract_one.res, value_op.res
        )

        rewriter.insert_op_before_matched_op(
            [
                minimum_value,
                minus_one,
                lhs_is_min_val,
                rhs_is_minus_one,
                is_underflow,
                zero,
                is_div_by_zero,
                introduce_poison,
                value_op,
                is_lhs_negative,
                is_rhs_negative,
                is_negative,
                remainder,
                is_remainder_not_zero,
                subtract_one,
                should_subtract_one,
                res_value_op,
            ]
        )
        return ((res_value_op.res, introduce_poison.res),)


arith_semantics: dict[type[Operation], OperationSemantics] = {
    arith.ConstantOp: ConstantSemantics(),
    arith.AddiOp: AddiSemantics(),
    arith.SubiOp: SubiSemantics(),
    arith.MuliOp: MuliSemantics(),
    arith.MulSIExtendedOp: MulSIExtendedSemantics(),
    arith.MulUIExtendedOp: MulUIExtendedSemantics(),
    arith.AndIOp: AndiSemantics(),
    arith.OrIOp: OriSemantics(),
    arith.XOrIOp: XoriSemantics(),
    arith.ShLIOp: ShliSemantics(),
    arith.DivSIOp: DivsiSemantics(),
    arith.DivUIOp: DivuiSemantics(),
    arith.RemSIOp: RemsiSemantics(),
    arith.RemUIOp: RemuiSemantics(),
    arith.ShRSIOp: ShrsiSemantics(),
    arith.ShRUIOp: ShruiSemantics(),
    arith.MaxSIOp: MaxsiSemantics(),
    arith.MaxUIOp: MaxuiSemantics(),
    arith.MinSIOp: MinsiSemantics(),
    arith.MinUIOp: MinuiSemantics(),
    arith.CmpiOp: CmpiSemantics(),
    arith.SelectOp: SelectSemantics(),
    arith.TruncIOp: TruncISemantics(),
    arith.ExtUIOp: ExtUISemantics(),
    arith.ExtSIOp: ExtSISemantics(),
    arith.CeilDivUIOp: CeilDivUISemantics(),
    arith.CeilDivSIOp: CeilDivSISemantics(),
    arith.FloorDivSIOp: FloorDivSISemantics(),
}
