from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence, cast
from xdsl.ir import Operation, SSAValue, Attribute
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.dialects.builtin import IntegerType

from xdsl_smt.semantics.semantics import AttributeSemantics, OperationSemantics

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


class AddiSemantics(SimplePurePoisonSemantics):
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


llvm_semantics: dict[type[Operation], OperationSemantics] = {
    llvm.AddOp: AddiSemantics(),
}
llvm_attribute_semantics: dict[type[Attribute], AttributeSemantics] = {
    llvm.OverflowAttr: OverflowAttrSemantics(),
}
