from dataclasses import dataclass

from xdsl.ir import SSAValue, ParametrizedAttribute, Attribute, Operation
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
)
from xdsl.utils.isattr import isattr

from xdsl.dialects.builtin import ModuleOp, ArrayAttr
from xdsl_smt.dialects.effects import memory_effect as mem_effect, ub_effect
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType
from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.dialects.effects import memory_effect as mem_effect
from xdsl_smt.dialects import (
    memory_dialect as mem,
    smt_utils_dialect as smt_utils,
    smt_dialect as smt,
    smt_bitvector_dialect as smt_bv,
)

new_state_type = smt_utils.PairType(mem.MemoryType(), BoolType())
new_pointer_type = smt_utils.PairType(mem.BlockIDType(), BitVectorType(64))


def get_memory(state: SSAValue, rewriter: PatternRewriter) -> SSAValue:
    """Get the memory state from the effect state."""
    memory_op = mem.GetMemoryOp(state)
    rewriter.insert_op_before_matched_op([memory_op])
    memory = memory_op.res
    memory.name_hint = "memory"
    return memory


def set_memory(
    state: SSAValue,
    memory: SSAValue,
    ub_condition: SSAValue | None,
    rewriter: PatternRewriter,
):
    new_state_op = mem.SetMemoryOp(state, memory)
    rewriter.insert_op_before_matched_op([new_state_op])
    new_state = new_state_op.res

    # If there is a UB condition, process it
    if ub_condition is not None:
        new_state_if_ub = ub_effect.TriggerOp(state)
        new_state_op = smt.IteOp(ub_condition, new_state_if_ub.res, new_state)
        rewriter.insert_op_before_matched_op([new_state_if_ub, new_state_op])
        new_state = new_state_op.res

    return new_state


def get_block_id_and_offset_from_pointer(
    pointer: SSAValue, rewriter: PatternRewriter
) -> tuple[SSAValue, SSAValue]:
    """Get the block ID and offset from a pointer value."""
    block_id = smt_utils.FirstOp(pointer, new_pointer_type)
    offset = smt_utils.SecondOp(pointer, new_pointer_type)
    rewriter.insert_op_before_matched_op([block_id, offset])
    block_id.res.name_hint = "block_id"
    offset.res.name_hint = "ptr_offset"
    return block_id.res, offset.res


def create_pointer(
    block_id: SSAValue, offset: SSAValue, rewriter: PatternRewriter
) -> SSAValue:
    """Create a pointer value from the block ID and offset."""
    pointer = smt_utils.PairOp(block_id, offset)
    rewriter.insert_op_before_matched_op([pointer])
    pointer.res.name_hint = "ptr"
    return pointer.res


def check_bounds(
    offset: SSAValue,
    block: SSAValue,
    rewriter: PatternRewriter,
) -> SSAValue:
    """Check if a pointer access is within the bounds of the memory."""

    # Get the block size
    block_size_op = mem.GetBlockSizeOp(block)
    rewriter.insert_op_before_matched_op([block_size_op])
    block_size = block_size_op.res
    block_size.name_hint = "block_size"

    # Check that the offset of the end bit is within the bounds
    offset_end_op = smt_bv.AddOp(offset, block_size_op.res)
    offset_in_bounds_op = smt_bv.UleOp(offset_end_op.res, block_size)
    rewriter.insert_op_before_matched_op([offset_end_op, offset_in_bounds_op])
    offset_in_bounds_op.res.name_hint = "offset_in_bounds"

    return offset_in_bounds_op.res


class LowerAlloc(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem_effect.AllocOp, rewriter: PatternRewriter):
        memory = get_memory(op.state, rewriter)

        # Get a fresh block ID
        id_op = mem.GetFreshBlockIDOp(memory)
        rewriter.insert_op_before_matched_op([id_op])
        memory = id_op.new_memory

        # Get the given block and set it as live with the given size
        get_block_op = mem.GetBlockOp(memory, id_op.res)
        block = get_block_op.res
        true_op = smt.ConstantBoolOp(True)
        set_live_op = mem.SetBlockLiveMarkerOp(block, true_op.res)
        block = set_live_op.res
        set_block_size = mem.SetBlockSizeOp(set_live_op.res, op.size)
        block = set_block_size.res
        rewriter.insert_op_before_matched_op(
            [get_block_op, true_op, set_live_op, set_block_size]
        )

        # Put it back to the memory
        new_memory_op = mem.SetBlockOp(block, memory, id_op.res)
        rewriter.insert_op_before_matched_op([new_memory_op])

        # Get a pointer to the block
        zero_index_op = smt_bv.ConstantOp(0, 64)
        rewriter.insert_op_before_matched_op([zero_index_op])
        new_pointer = create_pointer(id_op.res, zero_index_op.res, rewriter)

        # Update the memory in the state
        new_state = set_memory(op.state, new_memory_op.res, None, rewriter)

        # Replace the matched operation
        rewriter.replace_matched_op([], [new_state, new_pointer])


class LowerPointerOffset(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: mem_effect.OffsetPointerOp, rewriter: PatternRewriter
    ):
        # Get the block ID and offset from the pointer
        block_id, offset = get_block_id_and_offset_from_pointer(op.pointer, rewriter)

        # Add the offset to the given offset
        new_offset_op = smt_bv.AddOp(offset, op.offset)
        new_offset_op.res.name_hint = "ptr_offset"
        rewriter.insert_op_before_matched_op([new_offset_op])

        # Create the new pointer
        new_pointer = create_pointer(block_id, new_offset_op.res, rewriter)

        # Replace the matched operation
        rewriter.replace_matched_op([], [new_pointer])


class LowerRead(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem_effect.ReadOp, rewriter: PatternRewriter):
        # Unwrap the pointer and the state
        block_id, offset = get_block_id_and_offset_from_pointer(op.pointer, rewriter)
        memory_op = mem.GetMemoryOp(op.state)
        rewriter.insert_op_before_matched_op([memory_op])
        memory = memory_op.res

        # Get the memory block and bytes
        get_block_op = mem.GetBlockOp(memory, block_id)
        block = get_block_op.res
        get_block_bytes_op = mem.GetBlockBytesOp(block)
        bytes = get_block_bytes_op.res
        rewriter.insert_op_before_matched_op([get_block_op, get_block_bytes_op])

        # Check that the offset is within bounds
        offset_in_bounds = check_bounds(offset, block, rewriter)
        offset_not_in_bounds = smt.NotOp(offset_in_bounds)
        offset_not_in_bounds.res.name_hint = "offset_not_in_bounds"
        rewriter.insert_op_before_matched_op([offset_not_in_bounds])

        # Read the value in memory
        read_op = mem.ReadBytesOp(bytes, offset, op.res.type)
        rewriter.insert_op_before_matched_op([read_op])

        # Create the new state
        new_state = set_memory(op.state, memory, offset_not_in_bounds.res, rewriter)

        # Replace the matched operation
        rewriter.replace_matched_op([], [new_state, read_op.res])


class LowerWrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem_effect.WriteOp, rewriter: PatternRewriter):
        # Unwrap the pointer and the state
        block_id, offset = get_block_id_and_offset_from_pointer(op.pointer, rewriter)
        memory = get_memory(op.state, rewriter)

        # Get the memory block and bytes
        get_block_op = mem.GetBlockOp(memory, block_id)
        block = get_block_op.res
        get_block_bytes_op = mem.GetBlockBytesOp(block)
        bytes = get_block_bytes_op.res
        rewriter.insert_op_before_matched_op([get_block_op, get_block_bytes_op])

        # Check that the offset is within bounds, and update the ub flag
        offset_in_bounds = check_bounds(offset, block, rewriter)
        offset_not_in_bounds = smt.NotOp(offset_in_bounds)
        offset_not_in_bounds.res.name_hint = "offset_not_in_bounds"
        rewriter.insert_op_before_matched_op([offset_not_in_bounds])

        # Write the value in memory
        write_op = mem.WriteBytesOp(op.value, bytes, offset)
        bytes = write_op.res
        rewriter.insert_op_before_matched_op([write_op])

        # Update the bytes in the block and memory
        set_block_bytes_op = mem.SetBlockBytesOp(block, bytes)
        set_block_op = mem.SetBlockOp(set_block_bytes_op.res, memory, block_id)
        memory = set_block_op.res
        rewriter.insert_op_before_matched_op([set_block_bytes_op, set_block_op])

        # Create the new state
        new_state = set_memory(op.state, memory, offset_not_in_bounds.res, rewriter)

        # Replace the matched operation
        rewriter.replace_matched_op([], [new_state])


def recursively_convert_attr(attr: Attribute) -> Attribute:
    """
    Recursively convert an attribute to replace all references to the effect state
    into a pair between the ub flag and the memory.
    """
    if isinstance(attr, mem_effect.PointerType):
        return smt_utils.PairType(mem.BlockIDType(), BitVectorType(64))
    if isinstance(attr, ParametrizedAttribute):
        return type(attr).new(
            [recursively_convert_attr(param) for param in attr.parameters]
        )
    if isattr(attr, ArrayAttr):
        return ArrayAttr((recursively_convert_attr(value) for value in attr.data))
    return attr


class LowerGenericOp(RewritePattern):
    """
    Recursively lower all result types, attributes, and properties that reference
    effect states.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        for result in list(op.results):
            if (new_type := recursively_convert_attr(result.type)) != result.type:
                rewriter.replace_value_with_new_type(result, new_type)

        for region in op.regions:
            for block in region.blocks:
                for arg in list(block.args):
                    if (new_type := recursively_convert_attr(arg.type)) != arg.type:
                        rewriter.replace_value_with_new_type(arg, new_type)

        has_done_action = False
        for name, attr in op.attributes.items():
            if (new_attr := recursively_convert_attr(attr)) != attr:
                op.attributes[name] = new_attr
                has_done_action = True
        for name, attr in op.properties.items():
            if (new_attr := recursively_convert_attr(attr)) != attr:
                op.properties[name] = new_attr
                has_done_action = True
        if has_done_action:
            rewriter.handle_operation_modification(op)


@dataclass(frozen=True)
class LowerMemoryEffectsPass(ModulePass):
    name = "lower-memory-effects"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerAlloc(),
                    LowerPointerOffset(),
                    LowerRead(),
                    LowerWrite(),
                    LowerGenericOp(),
                ]
            )
        )
        walker.rewrite_module(op)
