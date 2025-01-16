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
from xdsl.rewriter import InsertPoint

from xdsl.dialects.builtin import ModuleOp, AnyArrayAttr
from xdsl_smt.dialects import (
    smt_utils_dialect as smt_utils,
    smt_dialect as smt,
    ub,
)
from xdsl_smt.traits.inhabitant import create_inhabitant


def recursively_convert_attr(attr: Attribute) -> Attribute:
    """
    Recursively convert an attribute to replace all references to the effect state
    into a pair between the ub flag and the memory.
    """
    if isattr(attr, ub.UBOrType[Attribute]):
        return smt_utils.PairType(attr.type, smt.BoolType())
    if isinstance(attr, ParametrizedAttribute):
        return type(attr).new(
            [recursively_convert_attr(param) for param in attr.parameters]
        )
    if isattr(attr, AnyArrayAttr):
        return AnyArrayAttr((recursively_convert_attr(value) for value in attr.data))
    return attr


class LowerGenericOp(RewritePattern):
    """
    Recursively lower all result types, attributes, and properties.
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


class LowerUBOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ub.UBOp, rewriter: PatternRewriter):
        assert isattr(op.res.type, ub.UBOrType[Attribute])
        inhabitant = create_inhabitant(op.res.type.type, rewriter)
        if inhabitant is None:
            raise ValueError(f"Type {op.res.type.type} does not have an inhabitant.")
        ub_flag = smt.ConstantBoolOp(True)
        pair = smt_utils.PairOp(inhabitant, ub_flag.res)
        rewriter.replace_matched_op([ub_flag, pair])


class LowerFromOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ub.FromOp, rewriter: PatternRewriter):
        assert isattr(op.res.type, ub.UBOrType[Attribute])
        ub_flag = smt.ConstantBoolOp(False)
        pair = smt_utils.PairOp(op.value, ub_flag.res)
        rewriter.replace_matched_op([ub_flag, pair])


class LowerMatchOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ub.MatchOp, rewriter: PatternRewriter):
        value = smt_utils.FirstOp(op.value)
        poison_flag = smt_utils.SecondOp(op.value)
        rewriter.insert_op_before_matched_op((value, poison_flag))
        value_terminator = op.value_terminator
        ub_terminator = op.ub_terminator
        rewriter.inline_block(
            op.value_region.block, InsertPoint.before(op), (value.res,)
        )
        rewriter.inline_block(op.ub_region.block, InsertPoint.before(op), ())
        results = list[SSAValue]()
        for val_val, val_ub in zip(value_terminator.rets, ub_terminator.rets):
            val = smt.IteOp(poison_flag.res, val_val, val_ub)
            results.append(val.res)
            rewriter.insert_op_before_matched_op(val)

        rewriter.erase_op(value_terminator)
        rewriter.erase_op(ub_terminator)

        rewriter.replace_matched_op([], results)


@dataclass(frozen=True)
class LowerUBToPairs(ModulePass):
    name = "lower-ub-to-pairs"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerUBOp(),
                    LowerFromOp(),
                    LowerMatchOp(),
                    LowerGenericOp(),
                ]
            )
        )
        walker.rewrite_module(op)
