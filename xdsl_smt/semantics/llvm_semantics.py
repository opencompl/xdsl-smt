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

        #TODO: fix this as test with disjoint flag still has boolean set as false
        disjoint_attr = attributes.get("disjointFlag")
        if disjoint_attr is None:
            disjoint_attr = rewriter.insert(smt.ConstantBoolOp(False)).res
        elif isinstance(disjoint_attr, Attribute):
            assert isinstance(disjoint_attr, llvm.UnitAttr)
            disjoint_attr = rewriter.insert(smt.ConstantBoolOp(True)).res
        else:
            disjoint_attr = cast(SSAValue[smt.BoolType], disjoint_attr)
        # Check if disjoint
        zero = rewriter.insert(smt_bv.ConstantOp(0, width)).res
        rhs_and_lhs = rewriter.insert(smt_bv.AndOp(lhs, rhs)).res
        has_no_carry_on = rewriter.insert(smt.EqOp(rhs_and_lhs, zero)).res
        has_carry_on = rewriter.insert(smt.NotOp(has_no_carry_on)).res
        
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

        # Correctly insert the ShlOp and retrieve its result
        res = rewriter.insert(smt_bv.ShlOp(lhs, rhs)).res

        # If the shift amount is greater than the width of the value, poison
        width_op = rewriter.insert(smt_bv.ConstantOp(width, width)).res
        poison_condition = rewriter.insert(smt_bv.UgtOp(rhs, width_op)).res

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

        # If the shift amount is greater than the width of the value, poison
        width_op = rewriter.insert(smt_bv.ConstantOp(width, width)).res
        poison_condition = rewriter.insert(smt_bv.UgtOp(rhs, width_op)).res

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

        # If the shift amount is greater than the width of the value, poison
        width_op = rewriter.insert(smt_bv.ConstantOp(width, width)).res
        poison_condition = rewriter.insert(smt_bv.UgtOp(rhs, width_op)).res

        return ((res, poison_condition),)

#  llvm.ShlOp: ShlSemantics(),
#    llvm.LShrOp: LshrSemantics(),
#    llvm.AShrOp: AshrSemantics(),


llvm_semantics: dict[type[Operation], OperationSemantics] = {
    llvm.AddOp: AddSemantics(),
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
