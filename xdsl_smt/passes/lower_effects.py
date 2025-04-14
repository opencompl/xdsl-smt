from dataclasses import dataclass
from xdsl.ir import Operation, Attribute, ParametrizedAttribute
from xdsl.utils.isattr import isattr
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)

from xdsl.dialects.builtin import ModuleOp, AnyArrayAttr
from xdsl_smt.dialects.smt_dialect import BoolType, ConstantBoolOp
from xdsl_smt.dialects.effects.effect import StateType
from xdsl_smt.dialects.effects.ub_effect import (
    ToBoolOp,
    TriggerOp,
)


def recursively_convert_attr(attr: Attribute) -> Attribute:
    if isinstance(attr, StateType):
        return BoolType()
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
    def match_and_rewrite(self, op: TriggerOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op(ConstantBoolOp(True))


class LowerToBoolOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ToBoolOp, rewriter: PatternRewriter):
        rewriter.replace_matched_op([], new_results=[op.state])


@dataclass(frozen=True)
class LowerEffectPass(ModulePass):
    """
    Lower the `effect` and `ub` dialects into the `smt` dialect.
    This pass should not be used in the presence of memory effects.
    """

    name = "lower-effects"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [LowerTriggerOp(), LowerToBoolOp(), LowerGenericOp()]
            )
        )
        walker.rewrite_module(op)
