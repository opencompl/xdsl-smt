from dataclasses import dataclass, field
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.pdl import (
    OperandOp,
    OperationOp,
    PatternOp,
    ReplaceOp,
    ResultOp,
    RewriteOp,
    TypeOp,
)
from xdsl.ir import Attribute, ErasedSSAValue, MLContext, Operation, SSAValue

from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.passes import ModulePass
from dialects.smt_dialect import AssertOp, CheckSatOp, DeclareConstOp, DistinctOp

from passes.arith_to_smt import ArithToSMT, convert_type


@dataclass
class PDLToSMTRewriteContext:
    pdl_op_to_op: dict[SSAValue, Operation] = field(default_factory=dict)


def _get_type_of_erased_type_value(value: SSAValue) -> Attribute:
    assert isinstance(value, ErasedSSAValue), "Error in rewriting logic"
    assert isinstance(
        (type_op := value.old_value.owner), TypeOp
    ), "Error in rewriting logic"
    type = type_op.constantType
    if type is None:
        raise Exception("Cannot handle non-constant types")
    return type


class PatternRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PatternOp, rewriter: PatternRewriter):
        rewriter.inline_block_before_matched_op(op.body.blocks[0])
        rewriter.insert_op_before_matched_op(CheckSatOp())
        rewriter.erase_matched_op()


class RewriteRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RewriteOp, rewriter: PatternRewriter):
        if op.body is not None:
            rewriter.inline_block_before_matched_op(op.body.blocks[0])
        rewriter.erase_matched_op()


class TypeRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TypeOp, rewriter: PatternRewriter):
        rewriter.erase_matched_op(safe_erase=False)


class OperandRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: OperandOp, rewriter: PatternRewriter):
        if op.valueType is None:
            raise Exception("Cannot handle non-typed operands")
        type = _get_type_of_erased_type_value(op.valueType)
        smt_type = convert_type(type)
        rewriter.replace_matched_op(DeclareConstOp(smt_type))


@dataclass
class OperationRewrite(RewritePattern):
    ctx: MLContext
    rewrite_context: PDLToSMTRewriteContext

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: OperationOp, rewriter: PatternRewriter):
        # Get the corresponding op definition from the context
        if op.opName is None:
            raise Exception("Cannot handle non-constant op names")
        op_def = self.ctx.get_op(op.opName.data)

        # Create the with the given operands and types
        result_types = [_get_type_of_erased_type_value(type) for type in op.typeValues]
        synthesized_op = op_def.create(
            operands=op.operandValues, result_types=result_types
        )

        # Cursed hack: we create a new module with that operation, and
        # we rewrite it with the arith_to_smt pass.
        rewrite_module = ModuleOp.from_region_or_ops([synthesized_op])
        ArithToSMT().apply(self.ctx, rewrite_module)
        last_op = rewrite_module.body.blocks[0].ops[-1]

        # Set the operation carrying the results in the context
        # FIXME: this does not work if the last operation does not return all results
        self.rewrite_context.pdl_op_to_op[op.op] = last_op
        rewriter.inline_block_before_matched_op(rewrite_module.body.blocks[0])

        rewriter.erase_matched_op(safe_erase=False)


@dataclass
class ReplaceRewrite(RewritePattern):
    rewrite_context: PDLToSMTRewriteContext

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReplaceOp, rewriter: PatternRewriter):
        assert isinstance(op.opValue, ErasedSSAValue)
        replaced_op = self.rewrite_context.pdl_op_to_op[op.opValue.old_value]
        if len(replaced_op.results) != 1:
            raise Exception("Cannot handle operations with multiple results")

        replacing_value: SSAValue
        # Replacing by values case
        if len(op.replValues) != 0:
            assert len(op.replValues) == 1
            replacing_value = op.replValues[0]
        # Replacing by operations case
        else:
            assert isinstance(op.replOperation, ErasedSSAValue)
            replacing_op = self.rewrite_context.pdl_op_to_op[op.replOperation.old_value]
            if len(replacing_op.results) != 1:
                raise Exception("Cannot handle operations with multiple results")
            replacing_value = replacing_op.results[0]

        distinct_op = DistinctOp(replacing_value, replaced_op.results[0])
        assert_op = AssertOp(distinct_op.res)
        rewriter.replace_matched_op([distinct_op, assert_op])


@dataclass
class ResultRewrite(RewritePattern):
    rewrite_context: PDLToSMTRewriteContext

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ResultOp, rewriter: PatternRewriter):
        assert isinstance(op.parent_, ErasedSSAValue)
        result = self.rewrite_context.pdl_op_to_op[op.parent_.old_value].results[
            op.index.value.data
        ]
        rewriter.replace_matched_op([], new_results=[result])


class PDLToSMT(ModulePass):
    name: str = "pdl-to-smt"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        rewrite_context = PDLToSMTRewriteContext({})
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PatternRewrite(),
                    RewriteRewrite(),
                    TypeRewrite(),
                    OperandRewrite(),
                    OperationRewrite(ctx, rewrite_context),
                    ReplaceRewrite(rewrite_context),
                    ResultRewrite(rewrite_context),
                ]
            )
        )
        walker.rewrite_module(op)
