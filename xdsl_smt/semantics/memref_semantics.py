from typing import Mapping, Sequence
from functools import reduce
import operator

from xdsl.utils.hints import isa
from xdsl.ir import Attribute, SSAValue, Operation
from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    MemRefType,
    NoneAttr,
    IntegerType,
    DYNAMIC_INDEX,
    ArrayAttr,
)
from xdsl.pattern_rewriter import PatternRewriter

from xdsl_smt.dialects.effects import ub_effect
import xdsl_smt.dialects.smt_bitvector_dialect as smt_bv
import xdsl_smt.dialects.smt_utils_dialect as smt_utils
import xdsl_smt.dialects.effects.memory_effect as mem_effect
import xdsl_smt.dialects.smt_dialect as smt
from xdsl_smt.dialects.smt_utils_dialect import PairType
from xdsl_smt.passes.lower_to_smt.smt_lowerer import SMTLowerer
from xdsl_smt.semantics.semantics import TypeSemantics, OperationSemantics


def get_value_and_poison(
    value: SSAValue, rewriter: PatternRewriter
) -> tuple[SSAValue, SSAValue]:
    """Extract the value and poison marker from an SSAValue pair."""
    first_op = smt_utils.FirstOp(value)
    second_op = smt_utils.SecondOp(value)
    rewriter.insert_op_before_matched_op([first_op, second_op])
    return first_op.res, second_op.res


def combine_poison_values(
    pairs: Sequence[SSAValue], rewriter: PatternRewriter
) -> tuple[list[SSAValue], SSAValue]:
    """
    Combine a sequence of value-poison pairs into a sequence of
    values, and a single poison marker.
    The resulting poison marker is true if any of the input poison markers are true.
    """
    unpacked = [get_value_and_poison(pair, rewriter) for pair in pairs]

    values = [value for value, _ in unpacked]
    poisons = [poison for _, poison in unpacked]
    if len(poisons) == 1:
        return values, poisons[0]

    poison = poisons[0]
    for other_poison in poisons[1:]:
        or_op = smt.OrOp(poison, other_poison)
        poison = or_op.result
        rewriter.insert_op_before_matched_op(or_op)

    return values, poison


class MemrefSemantics(TypeSemantics):
    def get_semantics(self, type: Attribute) -> Attribute:
        return smt_utils.PairType(mem_effect.PointerType(), smt.BoolType())


class AllocSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert isinstance(result := results[0], MemRefType)
        if result.layout != NoneAttr():
            raise ValueError("Cannot handle memrefs with layouts")
        if len(operands) != 0:
            raise ValueError("Cannot handle allocs with dynamic values")
        assert effect_state is not None

        # Create an allocation of the size of the memref
        alloc_size = reduce(operator.mul, result.get_shape(), 1)
        alloc_size_op = smt_bv.ConstantOp(alloc_size, 64)
        alloc_op = mem_effect.AllocOp(effect_state, alloc_size_op.res)

        # Alloc doesn't return poison
        poison_op = smt.ConstantBoolOp(False)
        poison_pointer_op = smt_utils.PairOp(alloc_op.pointer, poison_op.res)

        rewriter.insert_op_before_matched_op(
            [alloc_size_op, alloc_op, poison_op, poison_pointer_op]
        )
        return (poison_pointer_op.res,), alloc_op.new_state


class DeallocSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert effect_state is not None

        pointer, poison = get_value_and_poison(operands[0], rewriter)

        # Create a deallocation of the memref
        dealloc_op = mem_effect.DeallocOp(effect_state, pointer)

        # If the pointer is poisoned, trigger undefined behavior
        state_if_poison = ub_effect.TriggerOp(effect_state)
        new_state = smt.IteOp(poison, state_if_poison.res, dealloc_op.new_state)

        rewriter.insert_op_before_matched_op([dealloc_op, state_if_poison, new_state])

        return (), new_state.res


def offset_pointer_for_indices(
    pointer: SSAValue,
    indices: Sequence[SSAValue],
    sizes: Sequence[int],
    rewriter: PatternRewriter,
) -> tuple[SSAValue, SSAValue]:
    """
    Get the adress of an element in a memref, given the pointer to the memref, and
    a list of indices.
    Additionally return an SSAValue representing whether the indices are in bounds.
    """
    # Check that the memref can fit in 64 bits
    if reduce(operator.mul, sizes, 1) > 2**64:
        raise ValueError("Cannot handle memrefs larger than 2^64 bytes")

    zero_op = smt_bv.ConstantOp(0, 64)
    rewriter.insert_op_before_matched_op([zero_op])

    # Check that all indices are in bounds
    # This is sufficient to ensure that the pointer is in bounds, and that no
    # address computation is overflowing.
    in_bounds_op = smt.ConstantBoolOp(True)
    in_bounds = in_bounds_op.res
    rewriter.insert_op_before_matched_op(in_bounds_op)
    for index, size in zip(indices, sizes):
        nonnegative = smt_bv.SgeOp(index, zero_op.res)
        size_cst = smt_bv.ConstantOp(size, 64)
        less_size = smt_bv.SltOp(index, size_cst.res)
        in_bounds_dim = smt.AndOp(nonnegative.res, less_size.res)
        in_bounds_op = smt.AndOp(in_bounds, in_bounds_dim.result)
        in_bounds = in_bounds_op.result
        rewriter.insert_op_before_matched_op(
            [nonnegative, size_cst, less_size, in_bounds_dim, in_bounds_op]
        )

    # Compute the load offset
    offset_op = smt_bv.ConstantOp(0, 64)
    offset = offset_op.res
    rewriter.insert_op_before_matched_op(offset_op)
    for index, size in zip(indices, sizes):
        size_cst = smt_bv.ConstantOp(size, 64)
        mul_op = smt_bv.MulOp(offset, size_cst.res)
        add_op = smt_bv.AddOp(mul_op.res, index)
        offset = add_op.res
        rewriter.insert_op_before_matched_op([size_cst, mul_op, add_op])
    pointer_op = mem_effect.OffsetPointerOp(pointer, offset)
    rewriter.insert_op_before_matched_op(pointer_op)
    return (pointer_op.res, in_bounds)


class LoadSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert effect_state is not None

        # HACK to get the old operands
        assert isa(old_operands := attributes["__operand_types"], ArrayAttr)

        operand_values, poison = combine_poison_values(operands, rewriter)
        pointer, *indices = operand_values

        # Get the memref and its indices.
        # As the operand is now a pointer and not a memref anymore, we need to get
        # the old memref type.
        assert isa(memref_type := old_operands.data[0], MemRefType[Attribute])
        memref_size = memref_type.get_shape()
        memref_element = SMTLowerer.lower_type(memref_type.element_type)
        assert isa(memref_element, PairType[smt_bv.BitVectorType, smt.BoolType])

        # TODO: Support memrefs with layouts
        assert memref_type.layout == NoneAttr(), "Cannot handle memrefs with layouts"
        # TODO: Support memrefs with dynamic indices
        assert all(
            (index != DYNAMIC_INDEX for index in indices)
        ), "Cannot handle memrefs with dynamic indices"
        # TODO: Support memrefs with non-i8 element types
        assert memref_type.get_element_type() == IntegerType(
            8
        ), "Cannot handle memrefs with non-i8 element types"

        # Compute the load offset
        address, in_bounds = offset_pointer_for_indices(
            pointer, indices, memref_size, rewriter
        )

        read_op = mem_effect.ReadOp(effect_state, address, memref_element)
        rewriter.insert_op_before_matched_op([read_op])

        # Handle poison. If the pointer is poisoned, or any of the indices are poisoned,
        # or indices are not in bounds, undefined behavior is triggered.
        state_if_poison = ub_effect.TriggerOp(effect_state)
        not_in_bounds = smt.NotOp(in_bounds)
        ub_condition = smt.OrOp(poison, not_in_bounds.result)
        new_state = smt.IteOp(
            ub_condition.result, state_if_poison.res, read_op.new_state
        )
        rewriter.insert_op_before_matched_op(
            [state_if_poison, not_in_bounds, ub_condition, new_state]
        )

        return (read_op.res,), new_state.res


class StoreSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert effect_state is not None

        # HACK to get the old operands
        assert isa(old_operands := attributes["__operand_types"], ArrayAttr)

        (pointer, *indices), poison = combine_poison_values(operands[1:], rewriter)
        value = operands[0]

        # Get the memref and its indices.
        # As the operand is now a pointer and not a memref anymore, we need to get
        # the old memref type.
        assert isa(memref_type := old_operands.data[1], MemRefType[Attribute])
        memref_size = memref_type.get_shape()
        memref_element = SMTLowerer.lower_type(memref_type.element_type)
        assert isa(memref_element, PairType[smt_bv.BitVectorType, smt.BoolType])

        # TODO: Support memrefs with layouts
        assert memref_type.layout == NoneAttr(), "Cannot handle memrefs with layouts"
        # TODO: Support memrefs with dynamic indices
        assert all(
            (index != DYNAMIC_INDEX for index in indices)
        ), "Cannot handle memrefs with dynamic indices"
        # TODO: Support memrefs with non-i8 element types
        assert memref_type.get_element_type() == IntegerType(
            8
        ), "Cannot handle memrefs with non-i8 element types"

        # Compute the write offset
        address, in_bounds = offset_pointer_for_indices(
            pointer, indices, memref_size, rewriter
        )

        write_op = mem_effect.WriteOp(value, effect_state, address)
        rewriter.insert_op_before_matched_op([write_op])

        # Handle poison. If the pointer is poisoned, or any of the indices are poisoned,
        # undefined behavior is triggered.
        state_if_poison = ub_effect.TriggerOp(effect_state)
        not_in_bounds = smt.NotOp(in_bounds)
        ub_condition = smt.OrOp(poison, not_in_bounds.result)
        new_state = smt.IteOp(
            ub_condition.result, state_if_poison.res, write_op.new_state
        )
        rewriter.insert_op_before_matched_op(
            [state_if_poison, not_in_bounds, ub_condition, new_state]
        )

        return (), new_state.res


memref_semantics: dict[type[Operation], OperationSemantics] = {
    memref.AllocOp: AllocSemantics(),
    memref.DeallocOp: DeallocSemantics(),
    memref.LoadOp: LoadSemantics(),
    memref.StoreOp: StoreSemantics(),
}
