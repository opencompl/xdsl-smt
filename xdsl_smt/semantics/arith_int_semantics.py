from abc import abstractmethod, ABC
from xdsl_smt.semantics.semantics import OperationSemantics, TypeSemantics
from typing import Mapping, Sequence, cast
from xdsl.ir import SSAValue, Attribute, Operation, Region, Block
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.hints import isa
from xdsl.dialects.builtin import IntegerType, AnyIntegerAttr, IntegerAttr, FunctionType
from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_utils_dialect as smt_utils
from xdsl_smt.dialects.effects import ub_effect as smt_ub
from xdsl_smt.semantics.semantics import AttributeSemantics, RefinementSemantics
from xdsl.dialects.builtin import Signedness
from xdsl_smt.dialects import transfer
from xdsl.dialects import arith, pdl, comb
from xdsl_smt.passes.lower_to_smt.lower_to_smt import SMTLowerer

LARGE_ENOUGH_INT_TYPE = IntegerType(128, Signedness.UNSIGNED)
GENERIC_INT_WIDTH = 64


class IntAccessor(ABC):
    @abstractmethod
    def get_int_type(self, typ: Attribute) -> Attribute:
        ...

    @abstractmethod
    def get_payload(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        ...

    @abstractmethod
    def get_width(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        ...

    @abstractmethod
    def get_poison(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        ...

    @abstractmethod
    def get_int_max(
        self, smt_integer: SSAValue, typ: Attribute, rewriter: PatternRewriter
    ) -> SSAValue:
        ...

    @abstractmethod
    def get_packed_integer(
        self,
        payload: SSAValue,
        poison: SSAValue,
        width: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        ...


class FullIntAccessor(IntAccessor):
    def get_int_type(self, typ: Attribute) -> Attribute:
        inner_pair_type = smt_utils.PairType(smt_int.SMTIntType(), smt.BoolType())
        outer_pair_type = smt_utils.PairType(inner_pair_type, smt_int.SMTIntType())
        return outer_pair_type

    def get_payload(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        get_inner_pair_op = smt_utils.FirstOp(smt_integer)
        get_payload_op = smt_utils.FirstOp(get_inner_pair_op.res)
        rewriter.insert_op_before_matched_op([get_inner_pair_op, get_payload_op])
        return get_payload_op.res

    def get_width(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        get_width_op = smt_utils.SecondOp(smt_integer)
        rewriter.insert_op_before_matched_op([get_width_op])
        return get_width_op.res

    def get_poison(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        get_inner_pair_op = smt_utils.FirstOp(smt_integer)
        get_poison_op = smt_utils.SecondOp(get_inner_pair_op.res)
        rewriter.insert_op_before_matched_op([get_inner_pair_op, get_poison_op])
        return get_poison_op.res

    def get_int_max(
        self, smt_integer: SSAValue, typ: Attribute, rewriter: PatternRewriter
    ) -> SSAValue:
        assert isinstance(typ, IntegerType)
        int_max_op = smt_int.ConstantOp(2**typ.width.data, LARGE_ENOUGH_INT_TYPE)
        rewriter.insert_op_before_matched_op([int_max_op])
        return int_max_op.res

    def get_packed_integer(
        self,
        payload: SSAValue,
        poison: SSAValue,
        width: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        inner_pair = smt_utils.PairOp(payload, poison)
        outer_pair = smt_utils.PairOp(inner_pair.res, width)
        rewriter.insert_op_before_matched_op([inner_pair, outer_pair])
        return outer_pair.res


class PowEnabledIntAccessor(FullIntAccessor):
    def __init__(self, pow2_fun: SSAValue):
        self.pow2_fun = pow2_fun

    def get_int_max(
        self, smt_integer: SSAValue, typ: Attribute, rewriter: PatternRewriter
    ) -> SSAValue:
        if isinstance(typ, transfer.TransIntegerType) or isinstance(
            typ, smt_int.SMTIntType
        ):
            width = self.get_width(smt_integer, rewriter)
            int_max_op = smt.CallOp(self.pow2_fun, [width])
            rewriter.insert_op_before_matched_op([int_max_op])
            int_max = int_max_op.res[0]
        elif isinstance(typ, IntegerType):
            int_max_op = smt_int.ConstantOp(2**typ.width.data, LARGE_ENOUGH_INT_TYPE)
            rewriter.insert_op_before_matched_op([int_max_op])
            int_max = int_max_op.res
        else:
            assert False
        return int_max


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
        before_int_max = self.accessor.get_int_max(
            val_before, before_val.type, rewriter
        )
        # after_int_max = self.accessor.get_int_max(val_after, after_val.type, rewriter)
        int_max = before_int_max

        before_val_modulo_op = smt_int.ModOp(before_val, int_max)
        after_val_modulo_op = smt_int.ModOp(after_val, int_max)
        refinement = smt.EqOp(before_val_modulo_op.res, after_val_modulo_op.res)
        rewriter.insert_op_before_matched_op(
            [
                before_val_modulo_op,
                after_val_modulo_op,
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
        op = smt_int.ConstantOp(
            attribute.value.data, IntegerType(attribute.type.width.data)
        )
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
            literal = smt_int.ConstantOp(
                value_value.value.data, IntegerType(value_value.type.width.data)
            )
            assert isinstance(results[0], IntegerType)
            assert isinstance(results[0].width.data, int)
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
        rewriter.insert_op_before_matched_op([no_poison])
        result = self.accessor.get_packed_integer(
            ssa_attr, no_poison.res, int_max, rewriter
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
        lhs_int_max = self.accessor.get_int_max(lhs, results[0], rewriter)
        rhs_int_max = self.accessor.get_int_max(rhs, results[0], rewriter)
        eq_int_max_op = smt.EqOp(lhs_int_max, rhs_int_max)
        assert_eq_int_max_op = smt.AssertOp(eq_int_max_op.res)
        rewriter.insert_op_before_matched_op([eq_int_max_op, assert_eq_int_max_op])
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


def load_int_semantics(accessor: IntAccessor):
    semantics = {
        arith.Constant: IntConstantSemantics(accessor),
        arith.Addi: get_binary_ef_semantics(smt_int.AddOp)(accessor),
        arith.Subi: get_binary_ef_semantics(smt_int.SubOp)(accessor),
        arith.Muli: get_binary_ef_semantics(smt_int.MulOp)(accessor),
        arith.Cmpi: IntCmpiSemantics(accessor),
        arith.DivUI: get_div_semantics(smt_int.DivOp)(accessor),
        arith.RemUI: get_div_semantics(smt_int.ModOp)(accessor),
    }
    SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}
    types = {
        transfer.TransIntegerType: IntIntegerTypeSemantics(accessor),
        IntegerType: IntIntegerTypeSemantics(accessor),
    }
    SMTLowerer.type_lowerers = {**SMTLowerer.type_lowerers, **types}
    attribute_semantics: dict[type[Attribute], AttributeSemantics] = {
        IntegerAttr: IntIntegerAttrSemantics()
    }
    SMTLowerer.attribute_semantics = {
        **SMTLowerer.attribute_semantics,
        **attribute_semantics,
    }


def trigger_parametric_int(
    module_op: Operation,
):
    forbidden_ops = [arith.OrI.name, arith.ShLI.name, arith.DivSI.name]
    forbidden_ops += [o.name for o in comb.Comb.operations]
    use_parametric_int = True
    for inner_op in module_op.walk():
        if isinstance(inner_op, pdl.OperationOp):
            op_name = str(inner_op.opName).replace('"', "")
            if op_name in forbidden_ops:
                use_parametric_int = False
                break
        if isinstance(inner_op, pdl.ApplyNativeRewriteOp) or isinstance(
            inner_op, pdl.ApplyNativeConstraintOp
        ):
            use_parametric_int = False
            break
    return use_parametric_int


def insert_and_constraint_pow2(insert_point: InsertPoint):
    declare_pow_op = smt.DeclareFunOp(
        name="pow2",
        func_type=FunctionType.from_lists(
            inputs=[smt_int.SMTIntType()],
            outputs=[smt_int.SMTIntType()],
        ),
    )
    Rewriter.insert_op(declare_pow_op, insert_point)
    pow_in_type = smt_int.SMTIntType()
    # Forall x,y, x > y => pow2(x) > pow2(y)
    body = Region([Block(arg_types=[pow_in_type, pow_in_type])])
    assert isinstance(body.first_block, Block)
    x = body.first_block.args[0]
    y = body.first_block.args[1]
    x_gt_y_op = smt_int.GtOp(x, y)
    pow_x_op = smt.CallOp(declare_pow_op.ret, [x])
    pow_y_op = smt.CallOp(declare_pow_op.ret, [y])
    pow_x_gt_y_op = smt_int.GtOp(pow_x_op.res[0], pow_y_op.res[0])
    implies_op = smt.ImpliesOp(
        x_gt_y_op.res,
        pow_x_gt_y_op.res,
    )
    yield_op = smt.YieldOp(implies_op.res)
    assert isinstance(body.first_block, Block)
    body.first_block.add_ops(
        [x_gt_y_op, pow_x_op, pow_y_op, pow_x_gt_y_op, implies_op, yield_op]
    )
    forall_op0 = smt.ForallOp.create(result_types=[BoolType()], regions=[body])
    assert_op0 = smt.AssertOp(forall_op0.res)
    Rewriter.insert_op(forall_op0, insert_point)
    Rewriter.insert_op(assert_op0, insert_point)
    return declare_pow_op
