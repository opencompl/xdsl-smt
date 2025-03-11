"""Lower effects when only memory and ub effects are present."""

from dataclasses import dataclass
from xdsl.ir import Operation, Attribute, ParametrizedAttribute, SSAValue
from xdsl.utils.isattr import isattr
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.rewriter import InsertPoint
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)

from xdsl.dialects.builtin import ModuleOp, AnyArrayAttr
from xdsl_smt.dialects import (
    ub,
    smt_dialect as smt,
    memory_dialect as memory,
    smt_utils_dialect as smt_utils,
    smt_bitvector_dialect as smt_bv,
)
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType
from xdsl_smt.dialects.effects import effect, ub_effect, memory_effect as mem_effect

new_pointer_type = smt_utils.PairType(memory.BlockIDType(), BitVectorType(64))
new_state_type = ub.UBOrType(memory.MemoryType())


def recursively_convert_attr(attr: Attribute) -> Attribute:
    if isinstance(attr, effect.StateType):
        return ub.UBOrType(memory.MemoryType())
    if isinstance(attr, ParametrizedAttribute):
        return type(attr).new(
            [recursively_convert_attr(param) for param in attr.parameters]
        )
    if isattr(attr, AnyArrayAttr):
        return AnyArrayAttr((recursively_convert_attr(value) for value in attr.data))
    return attr


class LowerGenericOp(RewritePattern):
    """
    Recursively lower all result types, attributes, and properties that reference
    effect states.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        for result in op.results:
            if (new_type := recursively_convert_attr(result.type)) != result.type:
                rewriter.modify_value_type(result, new_type)

        for region in op.regions:
            for block in region.blocks:
                for arg in block.args:
                    if (new_type := recursively_convert_attr(arg.type)) != arg.type:
                        rewriter.modify_value_type(arg, new_type)

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


class LowerTriggerOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ub_effect.TriggerOp, rewriter: PatternRewriter):
        new_memory = rewriter.insert(ub.UBOp(memory.MemoryType())).res
        new_results = [new_memory]
        for result_type in op.result_types[1:]:
            new_results.append(rewriter.insert(ub.IrrelevantOp(result_type)).res)
        rewriter.replace_matched_op([], new_results)


class LowerToBoolOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ub_effect.ToBoolOp, rewriter: PatternRewriter):
        match_op = rewriter.insert(ub.MatchOp([op.state], [smt.BoolType()]))
        rewriter.replace_matched_op(match_op)

        rewriter.insertion_point = InsertPoint.at_end(match_op.value_region.block)
        true_val = rewriter.insert(smt.ConstantBoolOp(False)).res
        rewriter.insert(ub.YieldOp(true_val))

        rewriter.insertion_point = InsertPoint.at_end(match_op.ub_region.block)
        false_val = rewriter.insert(smt.ConstantBoolOp(True)).res
        rewriter.insert(ub.YieldOp(false_val))


def get_block_id_and_offset_from_pointer(
    pointer: SSAValue, rewriter: PatternRewriter
) -> tuple[SSAValue, SSAValue]:
    """Get the block ID and offset from a pointer value."""
    block_id = rewriter.insert(smt_utils.FirstOp(pointer, new_pointer_type)).res
    offset = rewriter.insert(smt_utils.SecondOp(pointer, new_pointer_type)).res
    block_id.name_hint = "block_id"
    offset.name_hint = "ptr_offset"
    return block_id, offset


def create_pointer(
    block_id: SSAValue, offset: SSAValue, rewriter: PatternRewriter
) -> SSAValue:
    """Create a pointer value from the block ID and offset."""
    pointer = smt_utils.PairOp(block_id, offset)
    rewriter.insert_op_before_matched_op([pointer])
    pointer.res.name_hint = "ptr"
    return pointer.res


class LowerPointerOffsetOp(RewritePattern):
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


class LowerAllocOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem_effect.AllocOp, rewriter: PatternRewriter):
        # Match for UB
        rewriter.insert(
            state_match := ub.MatchOp([op.state], [new_state_type, new_pointer_type])
        )
        rewriter.replace_matched_op([], state_match.results)

        # ub region
        rewriter.insertion_point = InsertPoint.at_end(state_match.ub_region.block)
        trigger_op = rewriter.insert(ub_effect.TriggerOp(op.state, (new_pointer_type,)))
        rewriter.insert(ub.YieldOp(*trigger_op.results))

        # value region
        rewriter.insertion_point = InsertPoint.at_end(state_match.value_region.block)
        memory_state = state_match.value_region.block.args[0]

        # Get a fresh block ID
        block_id, new_memory = rewriter.insert(
            memory.GetFreshBlockIDOp(memory_state)
        ).results

        # Get the given block and set it as live with the given size
        block = rewriter.insert(memory.GetBlockOp(new_memory, block_id)).res
        true = rewriter.insert(smt.ConstantBoolOp(True)).res
        new_block = rewriter.insert(memory.SetBlockLiveMarkerOp(block, true)).res
        new_block = rewriter.insert(memory.SetBlockSizeOp(new_block, op.size)).res

        # Put the block in memory, and get a pointer on it
        new_memory = rewriter.insert(
            memory.SetBlockOp(new_block, new_memory, block_id)
        ).res
        zero = rewriter.insert(smt_bv.ConstantOp(0, 64)).res
        pointer = create_pointer(block_id, zero, rewriter)

        # Update the state memory
        new_state = ub.FromOp(new_memory).res
        rewriter.insert(ub.YieldOp(new_state, pointer))


def check_bounds(
    offset: SSAValue,
    block: SSAValue,
    rewriter: PatternRewriter,
) -> SSAValue:
    """Check if a pointer access is within the bounds of the memory."""

    # Get the block size
    block_size = rewriter.insert(memory.GetBlockSizeOp(block)).res
    block_size.name_hint = "block_size"

    # Check that the offset of the end bit is within the bounds
    offset_end_op = rewriter.insert(smt_bv.AddOp(offset, block_size))
    offset_in_bounds_op = rewriter.insert(
        smt_bv.UleOp(offset_end_op.res, block_size)
    ).res
    offset_in_bounds_op.name_hint = "offset_in_bounds"

    return offset_in_bounds_op


class LowerReadOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem_effect.ReadOp, rewriter: PatternRewriter):
        # Match for UB
        rewriter.insert(
            state_match := ub.MatchOp([op.state], [new_state_type, op.res.type])
        )
        rewriter.replace_matched_op([], state_match.results)

        # ub region
        rewriter.insertion_point = InsertPoint.at_end(state_match.ub_region.block)
        trigger_op = rewriter.insert(ub_effect.TriggerOp(op.state, (op.res.type,)))
        rewriter.insert(ub.YieldOp(*trigger_op.results))

        # value region
        rewriter.insertion_point = InsertPoint.at_end(state_match.value_region.block)
        memory_state = state_match.value_region.block.args[0]

        # Unwrap the pointer and the state
        block_id, offset = get_block_id_and_offset_from_pointer(op.pointer, rewriter)

        # Get the memory block and bytes
        block = rewriter.insert(memory.GetBlockOp(memory_state, block_id)).res
        bytes = rewriter.insert(memory.GetBlockBytesOp(block)).res

        # Read the value in memory
        res = rewriter.insert(memory.ReadBytesOp(bytes, offset, op.res.type)).res

        # Check that the offset is within bounds
        offset_in_bounds = check_bounds(offset, block, rewriter)
        offset_not_in_bounds = rewriter.insert(smt.NotOp(offset_in_bounds)).res
        offset_not_in_bounds.name_hint = "offset_not_in_bounds"

        # Handle UB whenever the offset is not in bounds
        ub_state, ub_res = rewriter.insert(
            ub_effect.TriggerOp(op.state, (op.res.type,))
        ).results
        new_state = rewriter.insert(
            smt.IteOp(offset_not_in_bounds, ub_state, memory_state)
        ).res
        new_res = rewriter.insert(smt.IteOp(offset_not_in_bounds, ub_res, res)).res

        rewriter.insert(ub.YieldOp(new_state, new_res))


class LowerWriteOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem_effect.WriteOp, rewriter: PatternRewriter):
        # Match for UB
        rewriter.insert(state_match := ub.MatchOp([op.state], [new_state_type]))
        rewriter.replace_matched_op([], state_match.results)

        # ub region
        rewriter.insertion_point = InsertPoint.at_end(state_match.ub_region.block)
        trigger_op = rewriter.insert(ub_effect.TriggerOp(op.state))
        rewriter.insert(ub.YieldOp(*trigger_op.results))

        # value region
        rewriter.insertion_point = InsertPoint.at_end(state_match.value_region.block)
        memory_state = state_match.value_region.block.args[0]

        # Unwrap the pointer and the state
        block_id, offset = get_block_id_and_offset_from_pointer(op.pointer, rewriter)

        # Get the memory block and bytes
        block = rewriter.insert(memory.GetBlockOp(memory_state, block_id)).res
        bytes = rewriter.insert(memory.GetBlockBytesOp(block)).res

        # Read the value in memory
        new_state = rewriter.insert(memory.WriteBytesOp(bytes, offset, op.value)).res

        # Check that the offset is within bounds
        offset_in_bounds = check_bounds(offset, block, rewriter)
        offset_not_in_bounds = rewriter.insert(smt.NotOp(offset_in_bounds)).res
        offset_not_in_bounds.name_hint = "offset_not_in_bounds"

        # Handle UB whenever the offset is not in bounds
        ub_state = rewriter.insert(ub_effect.TriggerOp(op.state)).res
        new_state = rewriter.insert(
            smt.IteOp(offset_not_in_bounds, ub_state, new_state)
        ).res

        rewriter.insert(ub.YieldOp(new_state))


@dataclass(frozen=True)
class LowerMemoryAndUBEffectsPass(ModulePass):
    name = "lower-memory-and-ub-effects"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerTriggerOp(),
                    LowerToBoolOp(),
                    LowerGenericOp(),
                    LowerPointerOffsetOp(),
                    LowerAllocOp(),
                    LowerReadOp(),
                    LowerWriteOp(),
                ]
            )
        )
        walker.rewrite_module(op)
