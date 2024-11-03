from dataclasses import dataclass

from xdsl.ir import Attribute, ParametrizedAttribute, Operation, SSAValue
from xdsl.utils.isattr import isattr
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
)

from xdsl.dialects.builtin import ModuleOp, AnyArrayAttr
from xdsl_smt.dialects.effects import memory_effect as mem_effect
from xdsl_smt.dialects.effects.effect import StateType
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType
from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.dialects import (
    memory_dialect as mem,
    smt_utils_dialect as smt_utils,
    smt_dialect as smt,
    smt_bitvector_dialect as smt_bv,
)

from xdsl_smt.passes.lower_effects import LowerTriggerOp, LowerToBoolOp

new_state_type = smt_utils.PairType(mem.MemoryType(), BoolType())
new_pointer_type = smt_utils.PairType(mem.BlockIDType(), BitVectorType(64))


def get_ub_and_memory_from_state(
    state: SSAValue, rewriter: PatternRewriter
) -> tuple[SSAValue, SSAValue]:
    """Get the ub flag and memory from a state value."""
    memory = smt_utils.FirstOp(state, new_state_type)
    ub = smt_utils.SecondOp(state, new_state_type)
    rewriter.insert_op_before_matched_op([memory, ub])
    memory.res.name_hint = "memory"
    ub.res.name_hint = "ub_marker"
    return memory.res, ub.res


def create_state(memory: SSAValue, ub: SSAValue, rewriter: PatternRewriter) -> SSAValue:
    """Create a state value from the ub flag and memory state."""
    state = smt_utils.PairOp(memory, ub)
    rewriter.insert_op_before_matched_op([state])
    return state.res


def recursively_convert_attr(attr: Attribute) -> Attribute:
    """
    Recursively convert an attribute to replace all references to the effect state
    into a pair between the ub flag and the memory.
    """
    if isinstance(attr, StateType):
        return smt_utils.PairType(mem.MemoryType(), BoolType())
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


class LowerAlloc(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem_effect.AllocOp, rewriter: PatternRewriter):
        memory, ub = get_ub_and_memory_from_state(op.state, rewriter)

        # Get a fresh block ID
        id_op = mem.GetFreshBlockIDOp(memory)
        rewriter.insert_op_before_matched_op([id_op])

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
        new_memory_op = mem.SetBlockOp(memory, id_op.res, block)
        rewriter.insert_op_before_matched_op([new_memory_op])

        # Get a pointer to the block
        zero_index_op = smt_bv.ConstantOp(0, 64)
        new_pointer_op = smt_utils.PairOp(id_op.res, zero_index_op.res)
        rewriter.insert_op_before_matched_op([zero_index_op, new_pointer_op])

        # Get the new state
        new_state = create_state(new_memory_op.res, ub, rewriter)

        # Replace the matched operation
        rewriter.replace_matched_op([], [new_state, new_pointer_op.res])


@dataclass(frozen=True)
class LowerEffectWithMemoryPass(ModulePass):
    name = "lower-effects-with-memory"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [LowerTriggerOp(), LowerToBoolOp(), LowerGenericOp(), LowerAlloc()]
            )
        )
        walker.rewrite_module(op)
