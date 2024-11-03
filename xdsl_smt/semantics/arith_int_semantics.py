from abc import abstractmethod, ABC
from xdsl_smt.semantics.semantics import OperationSemantics, TypeSemantics
from typing import Mapping, Sequence, cast
from xdsl.ir import SSAValue, Attribute
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.hints import isa
from xdsl.dialects.builtin import IntegerType, AnyIntegerAttr, IntegerAttr
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_utils_dialect as smt_utils
from xdsl_smt.dialects.effects import ub_effect as smt_ub


def get_is_equal(lhs: SSAValue, rhs: SSAValue, rewriter: PatternRewriter) -> SSAValue:
    leop = smt_int.LeOp(lhs, rhs)
    geop = smt_int.GeOp(lhs, rhs)
    andop = smt.AndOp(leop.res, geop.res)
    rewriter.insert_op_before_matched_op([leop, geop, andop])
    return andop.res


def get_is_not_equal(
    lhs: SSAValue, rhs: SSAValue, rewriter: PatternRewriter
) -> SSAValue:
    ltop = smt_int.LtOp(lhs, rhs)
    gtop = smt_int.GtOp(lhs, rhs)
    orop = smt.OrOp(ltop.res, gtop.res)
    rewriter.insert_op_before_matched_op([ltop, gtop, orop])
    return orop.res


def get_int_max(width: int, rewriter: PatternRewriter) -> SSAValue:
    # 2 ^ width
    two = smt_int.ConstantOp(2)
    times: list[smt_int.ConstantOp | smt_int.MulOp] = [two]
    for _ in range(width - 1):
        ntimes = smt_int.MulOp(times[-1].res, two.res)
        times.append(ntimes)
    rewriter.insert_op_before_matched_op(times)
    return times[-1].res


def get_generic_modulo(
    x: SSAValue, int_max: SSAValue, rewriter: PatternRewriter
) -> SSAValue:
    # x % INT_MAX
    modulo0 = smt_int.ModOp(x, int_max)
    rewriter.insert_op_before_matched_op([modulo0])
    return modulo0.res


class IntIntegerTypeSemantics(TypeSemantics):
    def get_semantics(self, type: Attribute) -> Attribute:
        assert isinstance(type, IntegerType)
        inner_pair_type = smt_utils.PairType(smt_int.SMTIntType(), smt.BoolType())
        outer_pair_type = smt_utils.PairType(inner_pair_type, smt_int.SMTIntType())
        return outer_pair_type


class IntConstantSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        value_value = attributes["value"]
        assert isa(value_value, AnyIntegerAttr)
        assert len(results) == 1
        assert isa(results[0], IntegerType)
        literal = smt_int.ConstantOp(value_value.value.data)
        rewriter.insert_op_before_matched_op([literal])
        int_max = get_int_max(results[0].width.data, rewriter)
        modulo = get_generic_modulo(literal.res, int_max, rewriter)
        no_poison = smt.ConstantBoolOp.from_bool(False)
        inner_pair = smt_utils.PairOp(modulo, no_poison.res)
        outer_pair = smt_utils.PairOp(inner_pair.res, int_max)
        rewriter.insert_op_before_matched_op([no_poison, inner_pair, outer_pair])
        return ((outer_pair.res,), effect_state)


class AbsIntBinarySemantics(OperationSemantics, ABC):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        lhs = operands[0]
        rhs = operands[1]
        # Unpack
        lhs_get_inner_pair = smt_utils.FirstOp(lhs)
        rhs_get_inner_pair = smt_utils.FirstOp(rhs)
        lhs_get_int_max = smt_utils.SecondOp(lhs)
        rhs_get_int_max = smt_utils.SecondOp(rhs)
        # Check the int_max
        eq_op = smt.EqOp(lhs_get_int_max.res, rhs_get_int_max.res)
        assert_op = smt.AssertOp(eq_op.res)
        # Compute the payload
        lhs_get_payload = smt_utils.FirstOp(lhs_get_inner_pair.res)
        rhs_get_payload = smt_utils.FirstOp(rhs_get_inner_pair.res)
        rewriter.insert_op_before_matched_op(
            [
                lhs_get_inner_pair,
                rhs_get_inner_pair,
                lhs_get_int_max,
                rhs_get_int_max,
                eq_op,
                assert_op,
                lhs_get_payload,
                rhs_get_payload,
            ]
        )
        assert effect_state
        effect_state = self.get_ub(
            lhs_get_payload.res, rhs_get_payload.res, effect_state, rewriter
        )
        payload = self.get_payload_semantics(
            lhs_get_payload.res,
            rhs_get_payload.res,
            lhs_get_int_max.res,
            attributes,
            rewriter,
        )
        # Compute the poison
        lhs_get_poison = smt_utils.SecondOp(lhs_get_inner_pair.res)
        rhs_get_poison = smt_utils.SecondOp(rhs_get_inner_pair.res)
        rewriter.insert_op_before_matched_op([lhs_get_poison, rhs_get_poison])
        poison = self.get_poison(
            lhs_get_poison.res,
            rhs_get_poison.res,
            lhs_get_payload.res,
            rhs_get_payload.res,
            payload,
            rewriter,
        )
        # Pack
        inner_pair = smt_utils.PairOp(payload, poison)
        outer_pair = smt_utils.PairOp(inner_pair.res, lhs_get_int_max.res)
        rewriter.insert_op_before_matched_op([inner_pair, outer_pair])
        return ((outer_pair.res,), effect_state)

    @abstractmethod
    def get_ub(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        effect_state: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        ...

    @abstractmethod
    def get_poison(
        self,
        poison0: SSAValue,
        poison1: SSAValue,
        lhs: SSAValue,
        rhs: SSAValue,
        res: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        ...

    @abstractmethod
    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        int_max: SSAValue,
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> SSAValue:
        ...


class IntDivBasedSemantics(AbsIntBinarySemantics):
    @abstractmethod
    def _get_payload_semantics(
        self, lhs: SSAValue, rhs: SSAValue, rewriter: PatternRewriter
    ) -> SSAValue:
        ...

    def get_poison(
        self,
        poison0: SSAValue,
        poison1: SSAValue,
        lhs: SSAValue,
        rhs: SSAValue,
        res: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        or_poison = smt.OrOp(poison0, poison1)
        rewriter.insert_op_before_matched_op([or_poison])
        return or_poison.res

    def get_ub(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        effect_state: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        zero_op = smt_int.ConstantOp(0)
        rewriter.insert_op_before_matched_op([zero_op])
        is_div_by_zero = get_is_equal(rhs, zero_op.res, rewriter)
        trigger_ub = smt_ub.TriggerOp(effect_state)
        new_state = smt.IteOp(is_div_by_zero, trigger_ub.res, effect_state)
        rewriter.insert_op_before_matched_op([trigger_ub, new_state])
        return effect_state

    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        int_max: SSAValue,
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> SSAValue:
        return self._get_payload_semantics(lhs, rhs, rewriter)


class IntBinaryEFSemantics(AbsIntBinarySemantics):
    def get_ub(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        effect_state: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        return effect_state

    @abstractmethod
    def _get_payload_semantics(
        self, lhs: SSAValue, rhs: SSAValue, rewriter: PatternRewriter
    ) -> SSAValue:
        ...

    def get_poison(
        self,
        poison0: SSAValue,
        poison1: SSAValue,
        lhs: SSAValue,
        rhs: SSAValue,
        res: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        or_poison = smt.OrOp(poison0, poison1)
        rewriter.insert_op_before_matched_op([or_poison])
        return or_poison.res

    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        int_max: SSAValue,
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> SSAValue:
        payload = self._get_payload_semantics(lhs, rhs, rewriter)
        modulo = get_generic_modulo(payload, int_max, rewriter)
        return modulo


class IntCmpiSemantics(AbsIntBinarySemantics):
    def get_ub(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        effect_state: SSAValue,
        rewriter: PatternRewriter,
    ):
        return effect_state

    def get_poison(
        self,
        poison0: SSAValue,
        poison1: SSAValue,
        lhs: SSAValue,
        rhs: SSAValue,
        res: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        or_poison = smt.OrOp(poison0, poison1)
        rewriter.insert_op_before_matched_op([or_poison])
        return or_poison.res

    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        int_max: SSAValue,
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> SSAValue:
        integer_attr = cast(IntegerAttr[IntegerType], attributes["predicate"])
        match integer_attr.value.data:
            case 0:
                payload = get_is_equal(lhs, rhs, rewriter)
            case 1:
                payload = get_is_not_equal(lhs, rhs, rewriter)
            case 2:
                assert False
            case 3:
                assert False
            case 4:
                assert False
            case 5:
                assert False
            case 6:
                payload_op = smt_int.LtOp(lhs, rhs)
                rewriter.insert_op_before_matched_op([payload_op])
                payload = payload_op.res
            case 7:
                payload_op = smt_int.LeOp(lhs, rhs)
                rewriter.insert_op_before_matched_op([payload_op])
                payload = payload_op.res
            case 8:
                payload_op = smt_int.GtOp(lhs, rhs)
                rewriter.insert_op_before_matched_op([payload_op])
                payload = payload_op.res
            case 9:
                payload_op = smt_int.GeOp(lhs, rhs)
                rewriter.insert_op_before_matched_op([payload_op])
                payload = payload_op.res
            case _:
                assert False
        return payload


class IntAddiSemantics(IntBinaryEFSemantics):
    def _get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        payload_op = smt_int.AddOp(lhs, rhs)
        rewriter.insert_op_before_matched_op([payload_op])
        return payload_op.res


class IntSubiSemantics(IntBinaryEFSemantics):
    def _get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        payload_op = smt_int.SubOp(lhs, rhs)
        rewriter.insert_op_before_matched_op([payload_op])
        return payload_op.res


class IntMuliSemantics(IntBinaryEFSemantics):
    def _get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        payload_op = smt_int.MulOp(lhs, rhs)
        rewriter.insert_op_before_matched_op([payload_op])
        return payload_op.res


class IntDivUISemantics(IntDivBasedSemantics):
    def _get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        payload_op = smt_int.DivOp(lhs, rhs)
        rewriter.insert_op_before_matched_op([payload_op])
        return payload_op.res


class IntRemUISemantics(IntDivBasedSemantics):
    def _get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        payload_op = smt_int.ModOp(lhs, rhs)
        rewriter.insert_op_before_matched_op([payload_op])
        return payload_op.res
