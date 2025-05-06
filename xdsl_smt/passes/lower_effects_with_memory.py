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

from xdsl.dialects.builtin import ModuleOp, ArrayAttr
from xdsl_smt.dialects.effects import effect, ub_effect
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType
from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.dialects import (
    memory_dialect as mem,
    smt_utils_dialect as smt_utils,
    smt_dialect as smt,
)

new_state_type = smt_utils.PairType(mem.MemoryType(), BoolType())
new_pointer_type = smt_utils.PairType(mem.BlockIDType(), BitVectorType(64))


def get_memory_and_ub_from_state(
    state: SSAValue, rewriter: PatternRewriter
) -> tuple[SSAValue, SSAValue]:
    """Get the memory state and ub flag from the effect state."""
    memory_op = smt_utils.FirstOp(state)
    memory = memory_op.res
    memory.name_hint = "memory"

    ub_op = smt_utils.SecondOp(state)
    ub = ub_op.res
    ub.name_hint = "ub"

    rewriter.insert_op_before_matched_op([memory_op, ub_op])
    return memory, ub


def create_state(memory: SSAValue, ub: SSAValue, rewriter: PatternRewriter) -> SSAValue:
    """Create a state value from the ub flag and memory state."""
    state = smt_utils.PairOp(memory, ub)
    rewriter.insert_op_before_matched_op([state])
    state.res.name_hint = "state"
    return state.res


class LowerTriggerOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ub_effect.TriggerOp, rewriter: PatternRewriter):
        memory, _ = get_memory_and_ub_from_state(op.state, rewriter)
        new_ub = smt.ConstantBoolOp(True)
        rewriter.insert_op_before_matched_op([new_ub])
        new_state = create_state(memory, new_ub.res, rewriter)
        rewriter.replace_matched_op([], [new_state])


class LowerToBoolOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ub_effect.ToBoolOp, rewriter: PatternRewriter):
        _, ub = get_memory_and_ub_from_state(op.state, rewriter)
        rewriter.replace_matched_op([], new_results=[ub])


def recursively_convert_attr(attr: Attribute) -> Attribute:
    """
    Recursively convert an attribute to replace all references to the effect state
    into a pair between the ub flag and the memory.
    """
    if isinstance(attr, effect.StateType):
        return smt_utils.PairType(mem.MemoryType(), BoolType())
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
        for result in tuple(op.results):
            if (new_type := recursively_convert_attr(result.type)) != result.type:
                rewriter.replace_value_with_new_type(result, new_type)

        for region in op.regions:
            for block in region.blocks:
                for arg in tuple(block.args):
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


class LowerGetMemoryOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.GetMemoryOp, rewriter: PatternRewriter):
        memory, _ = get_memory_and_ub_from_state(op.state, rewriter)
        rewriter.replace_matched_op([], new_results=[memory])


class LowerSetMemoryOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.SetMemoryOp, rewriter: PatternRewriter):
        _, ub = get_memory_and_ub_from_state(op.state, rewriter)
        rewriter.replace_matched_op([], [create_state(op.memory, ub, rewriter)])


@dataclass(frozen=True)
class LowerEffectsWithMemoryPass(ModulePass):
    name = "lower-effects-with-memory"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerTriggerOp(),
                    LowerToBoolOp(),
                    LowerGenericOp(),
                    LowerGetMemoryOp(),
                    LowerSetMemoryOp(),
                ]
            )
        )
        walker.rewrite_module(op)
