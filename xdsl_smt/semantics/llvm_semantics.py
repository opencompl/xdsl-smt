from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence, cast
from xdsl.ir import Operation, SSAValue, Attribute
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.dialects.builtin import IntegerType, UnitAttr, IntegerAttr
from xdsl.utils.hints import isa

from xdsl_smt.semantics.semantics import AttributeSemantics, OperationSemantics
from xdsl_smt.semantics.builtin_semantics import IntegerAttrSemantics

from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_utils_dialect as smt_utils
import xdsl.dialects.llvm as llvm


@dataclass
class OverflowAttrSemanticsAdaptor:
    """
    An adaptor to manipulate integer overflow attribute semantics.
    """

    value: SSAValue[smt_utils.PairType[smt.BoolType, smt.BoolType]]

    @staticmethod
    def from_attribute(
        attribute: llvm.OverflowAttr, rewriter: PatternRewriter
    ) -> OverflowAttrSemanticsAdaptor:
        """
        Create an SSA value representing the integer overflow attribute.
        """
        nuw = rewriter.insert(
            smt.ConstantBoolOp(llvm.OverflowFlag.NO_UNSIGNED_WRAP in attribute.data)
        ).res
        nsw = rewriter.insert(
            smt.ConstantBoolOp(llvm.OverflowFlag.NO_SIGNED_WRAP in attribute.data)
        ).res
        res = rewriter.insert(smt_utils.PairOp(nuw, nsw)).res
        res = cast(SSAValue[smt_utils.PairType[smt.BoolType, smt.BoolType]], res)
        return OverflowAttrSemanticsAdaptor(res)

    def get_nuw_flag(self, rewriter: PatternRewriter) -> SSAValue[smt.BoolType]:
        """Get the unsigned wrap flag."""
        res = rewriter.insert(smt_utils.FirstOp(self.value)).res
        return cast(SSAValue[smt.BoolType], res)

    def get_nsw_flag(self, rewriter: PatternRewriter) -> SSAValue[smt.BoolType]:
        """Get the signed wrap flag."""
        res = rewriter.insert(smt_utils.SecondOp(self.value)).res
        return cast(SSAValue[smt.BoolType], res)


class OverflowAttrSemantics(AttributeSemantics):
    def get_semantics(
        self, attribute: Attribute, rewriter: PatternRewriter
    ) -> SSAValue:
        assert isinstance(attribute, llvm.OverflowAttr)
        return OverflowAttrSemanticsAdaptor.from_attribute(attribute, rewriter).value

    def get_unbounded_semantics(self, rewriter: PatternRewriter) -> SSAValue:
        return rewriter.insert(
            smt.DeclareConstOp(smt_utils.PairType(smt.BoolType(), smt.BoolType()))
        ).res


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
        result_poison = merge_poison.result
        rewriter.insert_op_before_matched_op(merge_poison)

    return values, result_poison


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
                pair = smt_utils.PairOp(value, poison.result)
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


class AddSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]
        res_type = results[0]
        assert isinstance(res_type, IntegerType)

        # Perform the addition
        res = rewriter.insert(smt_bv.AddOp(lhs, rhs)).res

        # Convert possible overflow attribute flag to a value
        overflow_attr = attributes["overflowFlags"]
        if isinstance(overflow_attr, Attribute):
            assert isinstance(overflow_attr, llvm.OverflowAttr)
            overflow_attr = OverflowAttrSemanticsAdaptor.from_attribute(
                overflow_attr, rewriter
            )
        else:
            overflow_attr = cast(
                SSAValue[smt_utils.PairType[smt.BoolType, smt.BoolType]], overflow_attr
            )
            overflow_attr = OverflowAttrSemanticsAdaptor(overflow_attr)

        # Handle nsw
        poison_condition = rewriter.insert(smt.ConstantBoolOp(False)).res
        has_nsw = overflow_attr.get_nsw_flag(rewriter)
        is_overflow = rewriter.insert(smt_bv.SaddOverflowOp(lhs, rhs)).res
        is_overflow_and_nsw = rewriter.insert(smt.AndOp(is_overflow, has_nsw)).result
        poison_condition = rewriter.insert(
            smt.OrOp(poison_condition, is_overflow_and_nsw)
        ).result

        # Handle nuw
        has_nuw = overflow_attr.get_nuw_flag(rewriter)
        is_overflow = rewriter.insert(smt_bv.UaddOverflowOp(lhs, rhs)).res
        is_overflow_and_nuw = rewriter.insert(smt.AndOp(is_overflow, has_nuw)).result
        poison_condition = rewriter.insert(
            smt.OrOp(poison_condition, is_overflow_and_nuw)
        ).result

        return ((res, poison_condition),)


# TODO add SsubOverflowOp and UsubOverflowOp
class SubSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]
        res_type = results[0]
        assert isinstance(res_type, IntegerType)

        # Perform the substraction
        res = rewriter.insert(smt_bv.SubOp(lhs, rhs)).res

        # Convert possible overflow attribute flag to a value
        overflow_attr = attributes["overflowFlags"]
        if isinstance(overflow_attr, Attribute):
            assert isinstance(overflow_attr, llvm.OverflowAttr)
            overflow_attr = OverflowAttrSemanticsAdaptor.from_attribute(
                overflow_attr, rewriter
            )
        else:
            overflow_attr = cast(
                SSAValue[smt_utils.PairType[smt.BoolType, smt.BoolType]], overflow_attr
            )
            overflow_attr = OverflowAttrSemanticsAdaptor(overflow_attr)

        # Handle nsw
        poison_condition = rewriter.insert(smt.ConstantBoolOp(False)).res
        has_nsw = overflow_attr.get_nsw_flag(rewriter)
        is_overflow = rewriter.insert(smt_bv.SaddOverflowOp(lhs, rhs)).res
        is_overflow_and_nsw = rewriter.insert(smt.AndOp(is_overflow, has_nsw)).result
        poison_condition = rewriter.insert(
            smt.OrOp(poison_condition, is_overflow_and_nsw)
        ).result

        # Handle nuw
        has_nuw = overflow_attr.get_nuw_flag(rewriter)
        is_overflow = rewriter.insert(smt_bv.UaddOverflowOp(lhs, rhs)).res
        is_overflow_and_nuw = rewriter.insert(smt.AndOp(is_overflow, has_nuw)).result
        poison_condition = rewriter.insert(
            smt.OrOp(poison_condition, is_overflow_and_nuw)
        ).result

        return ((res, poison_condition),)


class MulSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]
        res_type = results[0]
        assert isinstance(res_type, IntegerType)

        # Perform the addition
        res = rewriter.insert(smt_bv.MulOp(lhs, rhs)).res

        # Convert possible overflow attribute flag to a value
        overflow_attr = attributes["overflowFlags"]
        if isinstance(overflow_attr, Attribute):
            assert isinstance(overflow_attr, llvm.OverflowAttr)
            overflow_attr = OverflowAttrSemanticsAdaptor.from_attribute(
                overflow_attr, rewriter
            )
        else:
            overflow_attr = cast(
                SSAValue[smt_utils.PairType[smt.BoolType, smt.BoolType]], overflow_attr
            )
            overflow_attr = OverflowAttrSemanticsAdaptor(overflow_attr)

        # Handle nsw
        poison_condition = rewriter.insert(smt.ConstantBoolOp(False)).res
        has_nsw = overflow_attr.get_nsw_flag(rewriter)
        is_overflow = rewriter.insert(smt_bv.SmulOverflowOp(lhs, rhs)).res
        is_overflow_and_nsw = rewriter.insert(smt.AndOp(is_overflow, has_nsw)).result
        poison_condition = rewriter.insert(
            smt.OrOp(poison_condition, is_overflow_and_nsw)
        ).result

        # Handle nuw
        has_nuw = overflow_attr.get_nuw_flag(rewriter)
        is_overflow = rewriter.insert(smt_bv.UmulOverflowOp(lhs, rhs)).res
        is_overflow_and_nuw = rewriter.insert(smt.AndOp(is_overflow, has_nuw)).result
        poison_condition = rewriter.insert(
            smt.OrOp(poison_condition, is_overflow_and_nuw)
        ).result

        return ((res, poison_condition),)


class UdivSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]

        res = rewriter.insert(smt_bv.UDivOp(lhs, rhs)).res

        assert isinstance(lhs.type, smt_bv.BitVectorType)
        width = lhs.type.width.data

        # TODO: fix this -> see disjoint flag
        exact_attr = attributes.get("isExact")
        if exact_attr is None:
            exact_attr = rewriter.insert(smt.ConstantBoolOp(False)).res
        elif isinstance(exact_attr, Attribute):
            assert isinstance(exact_attr, UnitAttr)
            exact_attr = rewriter.insert(smt.ConstantBoolOp(True)).res
        else:
            exact_attr = cast(SSAValue[smt.BoolType], exact_attr)

        # Check if disjoint
        zero = rewriter.insert(smt_bv.ConstantOp(0, width)).res

        # get rhs modulo lhs
        modulo = rewriter.insert(smt_bv.URemOp(lhs, rhs)).res
        is_exact = rewriter.insert(smt.EqOp(modulo, zero)).res
        is_not_exact = rewriter.insert(smt.NotOp(is_exact)).result

        poison_condition = rewriter.insert(smt.AndOp(exact_attr, is_not_exact)).result

        return ((res, poison_condition),)


class SdivSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]

        res = rewriter.insert(smt_bv.UDivOp(lhs, rhs)).res

        assert isinstance(lhs.type, smt_bv.BitVectorType)
        width = lhs.type.width.data

        exact_attr = attributes.get("isExact")
        if exact_attr is None:
            exact_attr = rewriter.insert(smt.ConstantBoolOp(False)).res
        elif isinstance(exact_attr, Attribute):
            assert isinstance(exact_attr, UnitAttr)
            exact_attr = rewriter.insert(smt.ConstantBoolOp(True)).res
        else:
            exact_attr = cast(SSAValue[smt.BoolType], exact_attr)

        # Check if disjoint
        zero = rewriter.insert(smt_bv.ConstantOp(0, width)).res

        # get rhs modulo lhs
        modulo = rewriter.insert(smt_bv.SRemOp(lhs, rhs)).res
        is_exact = rewriter.insert(smt.EqOp(modulo, zero)).res
        is_not_exact = rewriter.insert(smt.NotOp(is_exact)).result

        poison_condition = rewriter.insert(smt.AndOp(exact_attr, is_not_exact)).result

        return ((res, poison_condition),)


class AndSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]

        # Perform the addition
        res = rewriter.insert(smt_bv.AndOp(lhs, rhs)).res

        return ((res, None),)


class XOrSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]

        # Perform the addition
        res = rewriter.insert(smt_bv.XorOp(lhs, rhs)).res

        return ((res, None),)


class OrSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]

        res = rewriter.insert(smt_bv.OrOp(lhs, rhs)).res

        assert isinstance(lhs.type, smt_bv.BitVectorType)
        width = lhs.type.width.data

        disjoint_attr = attributes.get("isDisjoint")
        print(attributes)
        # test case returns None :/
        if disjoint_attr is None:
            disjoint_attr = rewriter.insert(smt.ConstantBoolOp(False)).res
        elif isinstance(disjoint_attr, Attribute):
            assert isinstance(disjoint_attr, UnitAttr)
            disjoint_attr = rewriter.insert(smt.ConstantBoolOp(True)).res
        else:
            disjoint_attr = cast(SSAValue[smt.BoolType], disjoint_attr)
        # Check if disjoint
        zero = rewriter.insert(smt_bv.ConstantOp(0, width)).res
        rhs_and_lhs = rewriter.insert(smt_bv.AndOp(lhs, rhs)).res
        has_no_carry_on = rewriter.insert(smt.EqOp(rhs_and_lhs, zero)).res
        has_carry_on = rewriter.insert(smt.NotOp(has_no_carry_on)).result

        poison_condition = rewriter.insert(
            smt.AndOp(disjoint_attr, has_carry_on)
        ).result

        return ((res, poison_condition),)


class ShlSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]

        assert isinstance(lhs.type, smt_bv.BitVectorType)
        width = lhs.type.width.data

        # Convert possible overflow attribute flag to a value
        overflow_attr = attributes["overflowFlags"]
        if isinstance(overflow_attr, Attribute):
            assert isinstance(overflow_attr, llvm.OverflowAttr)
            overflow_attr = OverflowAttrSemanticsAdaptor.from_attribute(
                overflow_attr, rewriter
            )
        else:
            overflow_attr = cast(
                SSAValue[smt_utils.PairType[smt.BoolType, smt.BoolType]], overflow_attr
            )
            overflow_attr = OverflowAttrSemanticsAdaptor(overflow_attr)

        # Handle nsw
        poison_condition = rewriter.insert(smt.ConstantBoolOp(False)).res
        has_nsw = overflow_attr.get_nuw_flag(rewriter)
        left_shift = rewriter.insert(smt_bv.ShlOp(lhs, rhs)).res
        right_shift = rewriter.insert(smt_bv.AShrOp(left_shift, rhs)).res
        is_exact = rewriter.insert(smt.EqOp(lhs, right_shift)).res
        is_overflow = rewriter.insert(smt.NotOp(is_exact)).result
        is_overflow_and_nsw = rewriter.insert(smt.AndOp(is_overflow, has_nsw)).result
        poison_condition = rewriter.insert(
            smt.OrOp(poison_condition, is_overflow_and_nsw)
        ).result

        # Handle nuw
        has_nuw = overflow_attr.get_nuw_flag(rewriter)
        left_shift = rewriter.insert(smt_bv.ShlOp(lhs, rhs)).res
        right_shift = rewriter.insert(smt_bv.LShrOp(left_shift, rhs)).res
        is_exact = rewriter.insert(smt.EqOp(lhs, right_shift)).res
        is_overflow = rewriter.insert(smt.NotOp(is_exact)).result
        is_overflow_and_nuw = rewriter.insert(smt.AndOp(is_overflow, has_nuw)).result
        poison_condition = rewriter.insert(
            smt.OrOp(poison_condition, is_overflow_and_nuw)
        ).result

        # Correctly insert the ShlOp and retrieve its result
        res = rewriter.insert(smt_bv.ShlOp(lhs, rhs)).res

        # If the shift amount is greater than the width of the value, poison
        width_op = rewriter.insert(smt_bv.ConstantOp(width, width)).res
        invalid_width = rewriter.insert(smt_bv.UgtOp(rhs, width_op)).res
        poison_condition = rewriter.insert(
            smt.OrOp(poison_condition, invalid_width)
        ).result

        return ((res, poison_condition),)


class LshrSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]

        assert isinstance(lhs.type, smt_bv.BitVectorType)
        width = lhs.type.width.data

        # Correctly insert the ShlOp and retrieve its result
        res = rewriter.insert(smt_bv.LShrOp(lhs, rhs)).res

        exact_attr = attributes.get("isExact")
        if exact_attr is None:
            exact_attr = rewriter.insert(smt.ConstantBoolOp(False)).res
        elif isinstance(exact_attr, Attribute):
            assert isinstance(exact_attr, UnitAttr)
            exact_attr = rewriter.insert(smt.ConstantBoolOp(True)).res
        else:
            exact_attr = cast(SSAValue[smt.BoolType], exact_attr)

        # shift back and forth to check if its still the same
        right_shift = rewriter.insert(smt_bv.LShrOp(lhs, rhs)).res
        left_shift = rewriter.insert(smt_bv.ShlOp(right_shift, rhs)).res
        is_exact = rewriter.insert(smt.EqOp(lhs, left_shift)).res
        is_not_exact = rewriter.insert(smt.NotOp(is_exact)).result
        poison_condition = rewriter.insert(smt.AndOp(exact_attr, is_not_exact)).result

        # If the shift amount is greater than the width of the value, poison
        width_op = rewriter.insert(smt_bv.ConstantOp(width, width)).res
        wrong_length = rewriter.insert(smt_bv.UgtOp(rhs, width_op)).res

        poison_condition = rewriter.insert(
            smt.OrOp(poison_condition, wrong_length)
        ).result

        return ((res, poison_condition),)


class AshrSemantics(SimplePurePoisonSemantics):
    def get_pure_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[tuple[SSAValue, SSAValue | None]]:
        lhs = operands[0]
        rhs = operands[1]

        assert isinstance(lhs.type, smt_bv.BitVectorType)
        width = lhs.type.width.data

        # Correctly insert the Ashr and retrieve its result
        res = rewriter.insert(smt_bv.AShrOp(lhs, rhs)).res

        exact_attr = attributes.get("isExact")
        if exact_attr is None:
            exact_attr = rewriter.insert(smt.ConstantBoolOp(False)).res
        elif isinstance(exact_attr, Attribute):
            assert isinstance(exact_attr, UnitAttr)
            exact_attr = rewriter.insert(smt.ConstantBoolOp(True)).res
        else:
            exact_attr = cast(SSAValue[smt.BoolType], exact_attr)

        # shift back and forth to check if its still the same
        right_shift = rewriter.insert(smt_bv.LShrOp(lhs, rhs)).res
        left_shift = rewriter.insert(smt_bv.ShlOp(right_shift, rhs)).res
        is_exact = rewriter.insert(smt.EqOp(lhs, left_shift)).res
        is_not_exact = rewriter.insert(smt.NotOp(is_exact)).result
        poison_condition = rewriter.insert(smt.AndOp(exact_attr, is_not_exact)).result

        # If the shift amount is greater than the width of the value, poison
        width_op = rewriter.insert(smt_bv.ConstantOp(width, width)).res
        wrong_length = rewriter.insert(smt_bv.UgtOp(rhs, width_op)).res

        poison_condition = rewriter.insert(
            smt.OrOp(poison_condition, wrong_length)
        ).result

        return ((res, poison_condition),)


# TODO implement this, and add samesign flag
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
            assert isa(predicate_value, IntegerAttr)
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


llvm_semantics: dict[type[Operation], OperationSemantics] = {
    llvm.AddOp: AddSemantics(),
    llvm.SubOp: SubSemantics(),  # TODO add SsubOverflowOp and UsubOverflowOp
    llvm.MulOp: MulSemantics(),
    llvm.UDivOp: UdivSemantics(),
    llvm.SDivOp: SdivSemantics(),
    llvm.AndOp: AndSemantics(),
    llvm.XOrOp: XOrSemantics(),
    llvm.OrOp: OrSemantics(),
    llvm.ShlOp: ShlSemantics(),
    llvm.LShrOp: LshrSemantics(),
    llvm.AShrOp: AshrSemantics(),
}
llvm_attribute_semantics: dict[type[Attribute], AttributeSemantics] = {
    llvm.OverflowAttr: OverflowAttrSemantics(),
}
