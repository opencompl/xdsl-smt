from abc import abstractmethod, ABC
from xdsl_smt.semantics.semantics import OperationSemantics, TypeSemantics
from typing import Mapping, Sequence, cast
from xdsl.ir import SSAValue, Attribute, Operation, ErasedSSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.hints import isa
from xdsl.dialects.builtin import IntegerType, AnyIntegerAttr, IntegerAttr
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects.effects import ub_effect as smt_ub
from xdsl_smt.semantics.semantics import AttributeSemantics, RefinementSemantics
from xdsl_smt.dialects import transfer
from xdsl.dialects import arith, pdl, comb
from xdsl_smt.semantics.accessor import IntAccessor
from xdsl_smt.passes.pdl_to_smt_context import PDLToSMTRewriteContext

GENERIC_INT_WIDTH = 64


class IntIntegerTypeRefinementSemantics(RefinementSemantics):
    def __init__(self, accessor: IntAccessor):
        self.accessor = accessor

    def get_semantics(
        self,
        val_before: SSAValue,
        val_after: SSAValue,
        state_before: SSAValue,
        state_after: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        before_val = self.accessor.get_payload(val_before, rewriter)
        after_val = self.accessor.get_payload(val_after, rewriter)
        before_width = self.accessor.get_width(val_before, rewriter)
        after_width = self.accessor.get_width(val_after, rewriter)

        after_val_norm = self.accessor.get_signed_to_unsigned(
            after_val, after_width, rewriter
        )
        before_val_norm = self.accessor.get_signed_to_unsigned(
            before_val, before_width, rewriter
        )

        refinement = smt.EqOp(before_val_norm, after_val_norm)
        rewriter.insert_op_before_matched_op(
            refinement,
        )

        return refinement.res


class IntIntegerAttrSemantics(AttributeSemantics):
    def get_semantics(
        self, attribute: Attribute, rewriter: PatternRewriter
    ) -> SSAValue:
        if not isa(attribute, IntegerAttr[IntegerType]):
            raise Exception(f"Cannot handle semantics of {attribute}")
        op = smt_int.ConstantOp(attribute.value.data)
        rewriter.insert_op_before_matched_op(op)
        return op.res


class IntIntegerTypeSemantics(TypeSemantics):
    def __init__(self, accessor: IntAccessor):
        self.accessor = accessor

    def get_semantics(self, type: Attribute) -> Attribute:
        assert isinstance(type, IntegerType) or isinstance(
            type, transfer.TransIntegerType
        )
        int_type = self.accessor.get_int_type(type)
        return int_type


class GenericIntSemantics(OperationSemantics):
    def __init__(self, accessor: IntAccessor):
        self.accessor = accessor


class IntTruncISemantics(GenericIntSemantics):
    def __init__(self, accessor: IntAccessor, context: PDLToSMTRewriteContext):
        self.accessor = accessor
        self.context = context

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        operand = operands[0]
        payload = self.accessor.get_payload(operand, rewriter)
        poison = self.accessor.get_poison(operand, rewriter)
        assert isinstance(rewriter.current_operation, pdl.OperationOp)
        former_result = rewriter.current_operation.type_values[0]
        assert isinstance(former_result, ErasedSSAValue)
        pdl_type = former_result.old_value
        width = self.context.pdl_types_to_width[pdl_type]
        int_max = self.accessor.pow2_of(width, rewriter)
        modulo_op = smt_int.ModOp(payload, int_max)
        rewriter.insert_op_before_matched_op(
            [
                modulo_op,
            ]
        )
        packed_integer = self.accessor.get_packed_integer(
            modulo_op.res, poison, width, rewriter
        )
        return ((packed_integer,), effect_state)


class IntExtUISemantics(GenericIntSemantics):
    def __init__(self, accessor: IntAccessor, context: PDLToSMTRewriteContext):
        self.accessor = accessor
        self.context = context

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        operand = operands[0]
        payload = self.accessor.get_payload(operand, rewriter)
        poison = self.accessor.get_poison(operand, rewriter)
        assert isinstance(rewriter.current_operation, pdl.OperationOp)
        former_result = rewriter.current_operation.type_values[0]
        assert isinstance(former_result, ErasedSSAValue)
        pdl_type = former_result.old_value
        width = self.context.pdl_types_to_width[pdl_type]
        packed_integer = self.accessor.get_packed_integer(
            payload, poison, width, rewriter
        )
        return ((packed_integer,), effect_state)


class IntExtSISemantics(GenericIntSemantics):
    def __init__(self, accessor: IntAccessor, context: PDLToSMTRewriteContext):
        self.accessor = accessor
        self.context = context

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        operand = operands[0]
        payload = self.accessor.get_payload(operand, rewriter)
        poison = self.accessor.get_poison(operand, rewriter)
        width = self.accessor.get_width(operand, rewriter)
        assert isinstance(rewriter.current_operation, pdl.OperationOp)
        former_result = rewriter.current_operation.type_values[0]
        assert isinstance(former_result, ErasedSSAValue)
        pdl_type = former_result.old_value
        new_width = self.context.pdl_types_to_width[pdl_type]
        #
        signed_payload = self.accessor.get_unsigned_to_signed(payload, width, rewriter)
        unsigned_payload = self.accessor.get_signed_to_unsigned(
            signed_payload, width, rewriter
        )
        #
        packed_integer = self.accessor.get_packed_integer(
            unsigned_payload, poison, new_width, rewriter
        )
        return ((packed_integer,), effect_state)


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
            assert isinstance(value_value.type, IntegerType)
            assert isinstance(value_value.type.width.data, int)
            literal = smt_int.ConstantOp(value_value.value.data)
            assert isinstance(results[0], IntegerType)
            assert isinstance(results[0].width.data, int)
            width_op = smt_int.ConstantOp(results[0].width.data)
            width = width_op.res
            int_max_op = smt_int.ConstantOp(2 ** results[0].width.data)
            modulo = smt_int.ModOp(literal.res, int_max_op.res)
            rewriter.insert_op_before_matched_op(
                [literal, width_op, int_max_op, modulo]
            )
            ssa_attr = modulo.res
        else:
            width_op = smt_int.ConstantOp(GENERIC_INT_WIDTH)
            width = width_op.res
            int_max_op = smt_int.ConstantOp(2**GENERIC_INT_WIDTH)
            rewriter.insert_op_before_matched_op([width_op, int_max_op])
            ssa_attr = value_value

        no_poison = smt.ConstantBoolOp.from_bool(False)
        rewriter.insert_op_before_matched_op([no_poison])
        result = self.accessor.get_packed_integer(
            ssa_attr, no_poison.res, width, rewriter
        )

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
        # Constraint the width
        lhs_width = self.accessor.get_width(lhs, rewriter)
        rhs_width = self.accessor.get_width(rhs, rewriter)
        eq_width_op = smt.EqOp(lhs_width, rhs_width)
        assert_eq_width_op = smt.AssertOp(eq_width_op.res)
        rewriter.insert_op_before_matched_op([eq_width_op, assert_eq_width_op])
        # Compute the payload
        lhs_payload = self.accessor.get_payload(lhs, rewriter)
        rhs_payload = self.accessor.get_payload(rhs, rewriter)
        assert effect_state
        effect_state = self.get_ub(lhs_payload, rhs_payload, effect_state, rewriter)
        payload = self.get_payload_semantics(
            lhs_payload,
            rhs_payload,
            lhs_width,
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
            payload, poison, lhs_width, rewriter
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

    def get_ub(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        effect_state: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        return effect_state

    @abstractmethod
    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        width: SSAValue,
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
        zero_op = smt_int.ConstantOp(0)
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
        width: SSAValue,
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

    @abstractmethod
    def _get_payload_semantics(
        self, lhs: SSAValue, rhs: SSAValue, rewriter: PatternRewriter
    ) -> SSAValue:
        ...

    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        width: SSAValue,
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> SSAValue:
        payload = self._get_payload_semantics(lhs, rhs, rewriter)
        self.accessor.get_signed_to_unsigned(payload, width, rewriter)
        return payload


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
        width: SSAValue,
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


class IntOrISemantics(GenericIntBinarySemantics):
    def __init__(self, accessor: IntAccessor):
        super().__init__(accessor)

    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        width: SSAValue,
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> SSAValue:
        # TODO
        # andi = self.accessor.andi_of(width, lhs, rhs, rewriter)
        # xori = self.accessor.xori(width, lhs, rhs, rewriter)
        # ori_op = smt_int.AddOp(andi, xori)
        # rewriter.insert_op_before_matched_op([ori_op])
        # return ori_op.res
        raise NotImplementedError


class IntAndISemantics(GenericIntBinarySemantics):
    def __init__(self, accessor: IntAccessor):
        super().__init__(accessor)

    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        width: SSAValue,
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> SSAValue:
        andi = self.accessor.andi_of(width, lhs, rhs, rewriter)
        return andi


class IntXOrISemantics(GenericIntBinarySemantics):
    def __init__(self, accessor: IntAccessor):
        super().__init__(accessor)

    def get_payload_semantics(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        width: SSAValue,
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> SSAValue:
        raise NotImplementedError()


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


def trigger_parametric_int(
    module_op: Operation,
):
    forbidden_ops = [arith.OrI.name, arith.XOrI.name, arith.ShLI.name, arith.DivSI.name]
    forbidden_ops += [o.name for o in comb.Comb.operations]
    use_parametric_int = True
    for inner_op in module_op.walk():
        if isinstance(inner_op, pdl.OperationOp):
            op_name = str(inner_op.opName).replace('"', "")
            if op_name in forbidden_ops:
                use_parametric_int = False
                break
        # if isinstance(inner_op, pdl.ApplyNativeRewriteOp) or isinstance(
        #     inner_op, pdl.ApplyNativeConstraintOp
        # ):
        #     use_parametric_int = False
        #     break
    return use_parametric_int
