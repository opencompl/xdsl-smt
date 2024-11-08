from abc import abstractmethod, ABC
from xdsl_smt.semantics.semantics import OperationSemantics, TypeSemantics
from typing import Mapping, Sequence, cast
from xdsl.ir import SSAValue, Attribute
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.hints import isa
from xdsl.dialects.builtin import IntegerType, AnyIntegerAttr, IntegerAttr
from xdsl.dialects.builtin import IntegerType, AnyIntegerAttr, IntegerAttr, IndexType
from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_utils_dialect as smt_utils
from xdsl_smt.dialects.effects import ub_effect as smt_ub
from xdsl_smt.semantics.semantics import AttributeSemantics, RefinementSemantics
from xdsl_smt.dialects.effects import ub_effect
from xdsl.dialects.builtin import Signedness

LARGE_ENOUGH_INT_TYPE = IntegerType(128, Signedness.UNSIGNED)
GENERIC_INT_WIDTH = 64


class IntAccessor(ABC):
    @abstractmethod
    def get_int_type(self):
        ...

    @abstractmethod
    def get_int_max(self, smt_integer: SSAValue, rewriter: PatternRewriter):
        ...

    @abstractmethod
    def get_payload(self, smt_integer: SSAValue, rewriter: PatternRewriter):
        ...

    @abstractmethod
    def get_poison(self, smt_integer: SSAValue, rewriter: PatternRewriter):
        ...

    @abstractmethod
    def get_packed_integer(
        self,
        payload: SSAValue,
        poison: SSAValue,
        int_max: SSAValue,
        rewriter: PatternRewriter,
    ):
        ...


class FixedWidthIntAccessor(IntAccessor):
    def get_int_type(self):
        inner_pair_type = smt_utils.PairType(smt_int.SMTIntType(), smt.BoolType())
        outer_pair_type = smt_utils.PairType(inner_pair_type, smt_int.SMTIntType())
        return outer_pair_type

    def get_int_max(self, smt_integer: SSAValue, rewriter: PatternRewriter):
        get_int_max_op = smt_utils.SecondOp(smt_integer)
        rewriter.insert_op_before_matched_op([get_int_max_op])
        return get_int_max_op.res

    def get_payload(self, smt_integer: SSAValue, rewriter: PatternRewriter):
        get_inner_pair_op = smt_utils.FirstOp(smt_integer)
        get_payload_op = smt_utils.FirstOp(get_inner_pair_op.res)
        rewriter.insert_op_before_matched_op([get_inner_pair_op, get_payload_op])
        return get_payload_op.res

    def get_poison(self, smt_integer: SSAValue, rewriter: PatternRewriter):
        get_inner_pair_op = smt_utils.FirstOp(smt_integer)
        get_poison_op = smt_utils.SecondOp(get_inner_pair_op.res)
        rewriter.insert_op_before_matched_op([get_inner_pair_op, get_poison_op])
        return get_poison_op.res

    def get_packed_integer(
        self,
        payload: SSAValue,
        poison: SSAValue,
        int_max: SSAValue,
        rewriter: PatternRewriter,
    ):
        inner_pair = smt_utils.PairOp(payload, poison)
        outer_pair = smt_utils.PairOp(inner_pair.res, int_max)
        rewriter.insert_op_before_matched_op([inner_pair, outer_pair])
        return outer_pair.res


class IntIntegerAttrSemantics(AttributeSemantics):
    def get_semantics(
        self, attribute: Attribute, rewriter: PatternRewriter
    ) -> SSAValue:
        if not isa(attribute, IntegerAttr[IntegerType]):
            raise Exception(f"Cannot handle semantics of {attribute}")
        op = smt_int.ConstantOp(
            attribute.value.data, IntegerType(attribute.type.width.data)
        )
        rewriter.insert_op_before_matched_op(op)
        return op.res


class IntIntegerTypeSemantics(TypeSemantics):
    def __init__(self, accessor: IntAccessor):
        self.accessor = accessor

    def get_semantics(self, type: Attribute) -> Attribute:
        assert isinstance(type, IntegerType)
        int_type = self.accessor.get_int_type()
        return int_type


class GenericIntSemantics(OperationSemantics):
    def __init__(self, accessor: IntAccessor):
        self.accessor = accessor


class IntConstantSemantics(GenericIntSemantics):
    def __init__(self, accessor: IntAccessor):
        super().__init__(accessor)

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
            literal = smt_int.ConstantOp(
                value_value.value.data, IntegerType(value_value.type.width.data)
            )
            int_max_op = smt_int.ConstantOp(
                2 ** results[0].width.data, LARGE_ENOUGH_INT_TYPE
            )
            int_max = int_max_op.res
            modulo = smt_int.ModOp(literal.res, int_max_op.res)
            rewriter.insert_op_before_matched_op([literal, int_max_op, modulo])
            ssa_attr = modulo.res
        else:
            int_max_op = smt_int.ConstantOp(
                2**GENERIC_INT_WIDTH, LARGE_ENOUGH_INT_TYPE
            )
            int_max = int_max_op.res
            rewriter.insert_op_before_matched_op([int_max_op])
            ssa_attr = value_value

        no_poison = smt.ConstantBoolOp.from_bool(False)
        inner_pair = smt_utils.PairOp(ssa_attr, no_poison.res)
        outer_pair = smt_utils.PairOp(inner_pair.res, int_max)
        rewriter.insert_op_before_matched_op([no_poison, inner_pair, outer_pair])
        result = outer_pair.res

        return ((result,), effect_state)


class GenericIntBinarySemantics(GenericIntSemantics, ABC):
    """
    Generic semantics of binary operations on parametric integers.
    """

    def __init__(self, accessor: IntAccessor):
        super().__init__(accessor)

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
        lhs_int_max = self.accessor.get_int_max(lhs, rewriter)
        rhs_int_max = self.accessor.get_int_max(rhs, rewriter)
        # Compute the payload
        lhs_payload = self.accessor.get_payload(lhs, rewriter)
        rhs_payload = self.accessor.get_payload(rhs, rewriter)
        assert effect_state
        effect_state = self.get_ub(lhs_payload, rhs_payload, effect_state, rewriter)
        payload = self.get_payload_semantics(
            lhs_payload,
            rhs_payload,
            lhs_int_max,
            attributes,
            rewriter,
        )
        # Compute the poison
        lhs_poison = self.accessor.get_poison(lhs, rewriter)
        rhs_poison = self.accessor.get_poison(rhs, rewriter)
        poison = self.get_poison(
            lhs_poison,
            rhs_poison,
            lhs_payload,
            rhs_payload,
            payload,
            rewriter,
        )
        # Pack
        packed_integer = self.accessor.get_packed_integer(
            payload, poison, lhs_int_max, rewriter
        )
        return ((packed_integer,), effect_state)

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
    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        int_max: SSAValue,
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> SSAValue:
        ...


class IntDivBasedSemantics(GenericIntBinarySemantics):
    def __init__(self, accessor: IntAccessor):
        super().__init__(accessor)

    @abstractmethod
    def _get_payload_semantics(
        self, lhs: SSAValue, rhs: SSAValue, rewriter: PatternRewriter
    ) -> SSAValue:
        ...

    def get_ub(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        effect_state: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        zero_op = smt_int.ConstantOp(0, LARGE_ENOUGH_INT_TYPE)
        rewriter.insert_op_before_matched_op([zero_op])
        is_div_by_zero = smt.EqOp(rhs, zero_op.res)
        trigger_ub = smt_ub.TriggerOp(effect_state)
        new_state = smt.IteOp(is_div_by_zero.res, trigger_ub.res, effect_state)
        rewriter.insert_op_before_matched_op([is_div_by_zero, trigger_ub, new_state])
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


class IntBinaryEFSemantics(GenericIntBinarySemantics):
    """
    Semantics of binary operations on parametric integers which can not have an effect
    (Effect-Free).
    """

    def __init__(self, accessor: IntAccessor):
        super().__init__(accessor)

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

    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        int_max: SSAValue,
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> SSAValue:
        payload = self._get_payload_semantics(lhs, rhs, rewriter)
        modulo = smt_int.ModOp(payload, int_max)
        rewriter.insert_op_before_matched_op([modulo])
        return modulo.res


class IntCmpiSemantics(GenericIntBinarySemantics):
    def __init__(self, accessor: IntAccessor):
        super().__init__(accessor)

    def get_ub(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        effect_state: SSAValue,
        rewriter: PatternRewriter,
    ):
        return effect_state

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
                payload_op = smt.EqOp(lhs, rhs)
            case 1:
                payload_op = smt.DistinctOp(lhs, rhs)
            case 2:
                raise NotImplementedError()
            case 3:
                raise NotImplementedError()
            case 4:
                raise NotImplementedError()
            case 5:
                raise NotImplementedError()
            case 6:
                payload_op = smt_int.LtOp(lhs, rhs)
            case 7:
                payload_op = smt_int.LeOp(lhs, rhs)
            case 8:
                payload_op = smt_int.GtOp(lhs, rhs)
            case 9:
                payload_op = smt_int.GeOp(lhs, rhs)
            case _:
                assert False

        rewriter.insert_op_before_matched_op([payload_op])
        payload = payload_op.res

        return payload


def get_binary_ef_semantics(new_operation: type[smt_int.BinaryIntOp]):
    class OpSemantics(IntBinaryEFSemantics):
        def __init__(self, accessor: IntAccessor):
            super().__init__(accessor)

        def _get_payload_semantics(
            self,
            lhs: SSAValue,
            rhs: SSAValue,
            rewriter: PatternRewriter,
        ) -> SSAValue:
            payload_op = new_operation(lhs, rhs)
            assert not isinstance(payload_op, smt_int.DivOp)
            assert not isinstance(payload_op, smt_int.ModOp)
            rewriter.insert_op_before_matched_op([payload_op])
            return payload_op.res

    return OpSemantics


def get_div_semantics(new_operation: type[smt_int.BinaryIntOp]):
    class OpSemantics(IntDivBasedSemantics):
        def _get_payload_semantics(
            self,
            lhs: SSAValue,
            rhs: SSAValue,
            rewriter: PatternRewriter,
        ) -> SSAValue:
            payload_op = new_operation(lhs, rhs)
            assert isinstance(payload_op, smt_int.DivOp) or isinstance(
                payload_op, smt_int.ModOp
            )
            rewriter.insert_op_before_matched_op([payload_op])
            return payload_op.res

    return OpSemantics
