from abc import abstractmethod, ABC
from xdsl_smt.semantics.semantics import OperationSemantics, TypeSemantics
from typing import Mapping, Sequence, cast
from xdsl.ir import SSAValue, Attribute, Operation, ErasedSSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.hints import isa
from xdsl.dialects.builtin import IntegerType, AnyIntegerAttr, IntegerAttr, Signedness
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects.effects import ub_effect as smt_ub
from xdsl_smt.semantics.semantics import AttributeSemantics, RefinementSemantics
from xdsl_smt.dialects import transfer
from xdsl.dialects import arith, pdl, comb
from xdsl_smt.semantics.accessor import IntAccessor
from xdsl_smt.passes.pdl_to_smt_context import PDLToSMTRewriteContext
from xdsl_smt.dialects.effects import ub_effect
from xdsl.ir import OpResult


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
        before_poison = self.accessor.get_poison(val_before, rewriter)
        after_poison = self.accessor.get_poison(val_after, rewriter)
        before_width = self.accessor.get_width(val_before, rewriter)
        after_width = self.accessor.get_width(val_after, rewriter)

        after_val_norm = self.accessor.get_signed_to_unsigned(
            after_val, after_width, rewriter
        )
        before_val_norm = self.accessor.get_signed_to_unsigned(
            before_val, before_width, rewriter
        )
        eq_vals = smt.EqOp(before_val_norm, after_val_norm)
        not_before_poison = smt.NotOp(before_poison)
        not_after_poison = smt.NotOp(after_poison)
        not_poison_eq = smt.AndOp(eq_vals.res, not_after_poison.res)
        refinement_integer = smt.ImpliesOp(not_before_poison, not_poison_eq.res)
        rewriter.insert_op_before_matched_op(
            [
                not_before_poison,
                not_after_poison,
                eq_vals,
                not_poison_eq,
                refinement_integer,
            ]
        )

        # With UB, our refinement is: ub_before \/ (not ub_after /\ integer_refinement)
        ub_before_bool = ub_effect.ToBoolOp(state_before)
        ub_after_bool = ub_effect.ToBoolOp(state_after)
        not_ub_after = smt.NotOp(ub_after_bool.res)
        not_ub_before_case = smt.AndOp(not_ub_after.res, refinement_integer.res)
        refinement = smt.OrOp(ub_before_bool.res, not_ub_before_case.res)
        rewriter.insert_op_before_matched_op(
            [
                ub_before_bool,
                ub_after_bool,
                not_ub_after,
                not_ub_before_case,
                refinement,
            ]
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


class IntSelectSemantics(GenericIntSemantics):
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
        cond_val = self.accessor.get_payload(operands[0], rewriter)
        cond_poi = self.accessor.get_poison(operands[0], rewriter)
        tr_val = self.accessor.get_payload(operands[1], rewriter)
        tr_poi = self.accessor.get_poison(operands[1], rewriter)
        fls_val = self.accessor.get_payload(operands[2], rewriter)
        fls_poi = self.accessor.get_poison(operands[2], rewriter)
        fls_wid = self.accessor.get_width(operands[2], rewriter)

        # Get the resulting value depending on the condition
        res_val = smt.IteOp(cond_val, tr_val, fls_val)
        br_poi = smt.IteOp(cond_val, tr_poi, fls_poi)

        # If the condition is poison, the result is poison
        res_poi = smt.IteOp(cond_poi, cond_poi, br_poi.res)

        rewriter.insert_op_before_matched_op([res_val, br_poi, res_poi])

        res = self.accessor.get_packed_integer(
            res_val.res, res_poi.res, fls_wid, rewriter
        )

        return ((res,), effect_state)


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
            width_attr = value_value.type.width.data
            literal = smt_int.ConstantOp(
                value=value_value.value.data,
            )
            assert isinstance(results[0], IntegerType)
            assert isinstance(results[0].width.data, int)
            width_res = results[0].width.data
            assert width_res == width_attr
            width_op = smt_int.ConstantOp(value=width_res)
            width = width_op.res
            int_max_op = smt_int.ConstantOp(
                2 ** results[0].width.data,
            )
            modulo = smt_int.ModOp(literal.res, int_max_op.res)
            rewriter.insert_op_before_matched_op(
                [literal, width_op, int_max_op, modulo]
            )
            ssa_attr = modulo.res
        else:
            if isinstance(results[0], IntegerType):
                width_op = smt_int.ConstantOp(results[0].width.data)
            elif isinstance(results[0], transfer.TransIntegerType):
                width_op = smt_int.ConstantOp(GENERIC_INT_WIDTH)
            else:
                assert False
            width = width_op.res
            rewriter.insert_op_before_matched_op([width_op])
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
        if isinstance(attributes["predicate"], IntegerAttr):
            predicate_attr = attributes["predicate"]
        elif isinstance(attributes["predicate"], OpResult):
            op = attributes["predicate"].op
            assert isinstance(op, smt_int.ConstantOp)
            predicate_attr = op.value
        else:
            assert False
        # Constants definition
        one_const = smt_int.ConstantOp(1)
        zero_const = smt_int.ConstantOp(0)
        rewriter.insert_op_before_matched_op([one_const, zero_const])
        # Comparison
        predicate = predicate_attr.value.data
        match predicate:
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

        cast_op = smt.IteOp(payload_op.res, one_const.res, zero_const.res)
        rewriter.insert_op_before_matched_op([payload_op, cast_op])
        result = cast_op.res

        return result


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
        if isinstance(inner_op, pdl.ApplyNativeConstraintOp):
            constraint_name = str(inner_op.constraint_name).replace('"', "")
            if constraint_name == "is_minus_one":
                use_parametric_int = False
                break
        if isinstance(inner_op, pdl.ApplyNativeRewriteOp):
            constraint_name = str(inner_op.constraint_name).replace('"', "")
            if constraint_name == "get_width":
                use_parametric_int = False
                break
    return use_parametric_int
