from abc import abstractmethod, ABC
from xdsl_smt.semantics.semantics import OperationSemantics
from typing import Mapping, Sequence
from xdsl.ir import SSAValue, Attribute
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.hints import isa
from xdsl.dialects.builtin import IntegerType, AnyIntegerAttr
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_utils_dialect as smt_utils


def get_int_max(width: int):
    # 2 ^ width
    two = smt_int.ConstantOp(2)
    times: list[smt_int.ConstantOp | smt_int.MulOp] = [two]
    for _ in range(width - 1):
        ntimes = smt_int.MulOp(times[-1].res, two.res)
        times.append(ntimes)
    return times


def get_generic_modulo(x: SSAValue, int_max: SSAValue):
    # x % INT_MAX
    modulo0 = smt_int.ModOp(x, int_max)
    return [modulo0]


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
        int_max = get_int_max(results[0].width.data)
        assert len(int_max) > 0
        modulo = get_generic_modulo(literal.res, int_max[-1].res)
        no_poison = smt.ConstantBoolOp.from_bool(False)
        inner_pair = smt_utils.PairOp(modulo[-1].res, no_poison.res)
        outer_pair = smt_utils.PairOp(inner_pair.res, int_max[-1].res)
        rewriter.insert_op_before_matched_op(
            int_max + modulo + [literal, no_poison, inner_pair, outer_pair]
        )
        return ((outer_pair.res,), effect_state)


class IntBinarySemantics(OperationSemantics, ABC):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        #
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
        modular_payload_semantics = self.get_modular_payload_semantics(
            lhs_get_payload.res,
            rhs_get_payload.res,
            lhs_get_int_max.res,
        )
        assert modular_payload_semantics
        payload = modular_payload_semantics[-1].res
        # Compute the poison
        lhs_get_poison = smt_utils.SecondOp(lhs_get_inner_pair.res)
        rhs_get_poison = smt_utils.SecondOp(rhs_get_inner_pair.res)
        get_poison = self.get_poison(
            lhs_get_poison.res,
            rhs_get_poison.res,
            lhs_get_payload.res,
            rhs_get_payload.res,
            payload,
        )
        assert get_poison
        poison = get_poison[-1].res
        # Pack
        inner_pair = smt_utils.PairOp(payload, poison)
        outer_pair = smt_utils.PairOp(inner_pair.res, lhs_get_int_max.res)
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
                lhs_get_poison,
                rhs_get_poison,
            ]
            + modular_payload_semantics
            + get_poison
            + [
                inner_pair,
                outer_pair,
            ]
        )
        return ((outer_pair.res,), effect_state)

    @abstractmethod
    def get_payload_semantics(
        self, lhs: SSAValue, rhs: SSAValue
    ) -> smt_int.BinaryIntOp:
        ...

    def get_poison(
        self,
        poison0: SSAValue,
        poison1: SSAValue,
        lhs: SSAValue,
        rhs: SSAValue,
        res: SSAValue,
    ):
        or_poison = smt.OrOp(poison0, poison1)
        return [or_poison]

    def get_modular_payload_semantics(
        self, lhs: SSAValue, rhs: SSAValue, int_max: SSAValue
    ):
        payload_op = self.get_payload_semantics(lhs, rhs)
        modulo = get_generic_modulo(payload_op.res, int_max)
        return [payload_op] + modulo


class IntAddSemantics(IntBinarySemantics):
    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
    ):
        payload_op = smt_int.AddOp(lhs, rhs)
        return payload_op


class IntSubSemantics(IntBinarySemantics):
    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
    ):
        payload_op = smt_int.SubOp(lhs, rhs)
        return payload_op


class IntMulSemantics(IntBinarySemantics):
    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
    ):
        payload_op = smt_int.MulOp(lhs, rhs)
        return payload_op
