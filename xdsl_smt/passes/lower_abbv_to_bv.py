from dataclasses import dataclass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
)
from xdsl.passes import ModulePass
from xdsl.context import Context

from xdsl_smt.dialects import (
    smt_bitvector_dialect as smt_bv,
    ab_bitvector_dialect as ab_bv,
)
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation

from xdsl_smt.passes.dead_code_elimination import DeadCodeElimination


class ConstantOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ab_bv.ConstantOp, rewriter: PatternRewriter):
        if not isinstance(op.width.owner, ab_bv.ConstantBitWidthOp):
            return

        value = op.value.data
        width = op.width.owner.value.data
        cst = rewriter.insert(smt_bv.ConstantOp(value, width)).res
        rewriter.replace_op(op, ab_bv.FromFixedBitWidthOp(cst))


class FromToFixedBitWidthLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ab_bv.ToFixedBitWidthOp, rewriter: PatternRewriter):
        if not isinstance(op.operand.owner, ab_bv.FromFixedBitWidthOp):
            return
        if op.result.type != op.operand.owner.operand.type:
            raise ValueError("Mismatched types in From/ToFixedBitWidth lowering")
        rewriter.replace_op(op, [], new_results=[op.operand.owner.operand])


class GetBitWidthFromFixedBitWidthLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ab_bv.GetBitWidthOp, rewriter: PatternRewriter):
        if not isinstance(op.operand.owner, ab_bv.FromFixedBitWidthOp):
            return

        type = op.operand.owner.operand.type
        assert isinstance(type, smt_bv.BitVectorType)
        rewriter.replace_op(op, ab_bv.ConstantBitWidthOp(type.width.data))


def unary_op_lowering_factory(
    op_cls: type[ab_bv.UnaryBVOp], smt_bv_op_cls: type[smt_bv.UnaryBVOp]
) -> type[RewritePattern]:
    @dataclass(frozen=True)
    class UnaryOpLowering(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, op_cls):
                return
            if not isinstance(op.arg.owner, ab_bv.FromFixedBitWidthOp):
                return
            res = rewriter.insert(smt_bv_op_cls(op.arg.owner.operand)).res
            rewriter.replace_op(op, ab_bv.FromFixedBitWidthOp(res))

    return UnaryOpLowering


def binary_op_lowering_factory(
    op_cls: type[ab_bv.BinaryBVOp], smt_bv_op_cls: type[smt_bv.BinaryBVOp]
) -> type[RewritePattern]:
    @dataclass(frozen=True)
    class BinaryOpLowering(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, op_cls):
                return
            if not isinstance(op.lhs.owner, ab_bv.FromFixedBitWidthOp):
                return
            if not isinstance(op.rhs.owner, ab_bv.FromFixedBitWidthOp):
                return
            res = rewriter.insert(
                smt_bv_op_cls(op.lhs.owner.operand, op.rhs.owner.operand)
            ).res
            rewriter.replace_op(op, ab_bv.FromFixedBitWidthOp(res))

    return BinaryOpLowering


class CmpOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ab_bv.CmpOp, rewriter: PatternRewriter):
        if not isinstance(op.lhs.owner, ab_bv.FromFixedBitWidthOp):
            return
        if not isinstance(op.rhs.owner, ab_bv.FromFixedBitWidthOp):
            return
        rewriter.replace_op(
            op, smt_bv.CmpOp(op.pred, op.lhs.owner.operand, op.rhs.owner.operand)
        )


def unary_pred_op_lowering_factory(
    op_cls: type[ab_bv.UnaryPredBVOp], smt_bv_op_cls: type[smt_bv.UnaryPredBVOp]
) -> type[RewritePattern]:
    @dataclass(frozen=True)
    class UnaryPredOpLowering(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, op_cls):
                return
            if not isinstance(op.operand.owner, ab_bv.FromFixedBitWidthOp):
                return
            rewriter.replace_op(op, smt_bv_op_cls(op.operand.owner.operand))

    return UnaryPredOpLowering


class ConcatLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ab_bv.ConcatOp, rewriter: PatternRewriter):
        if not isinstance(op.lhs.owner, ab_bv.FromFixedBitWidthOp):
            return
        if not isinstance(op.rhs.owner, ab_bv.FromFixedBitWidthOp):
            return
        res = rewriter.insert(
            smt_bv.ConcatOp(op.lhs.owner.operand, op.rhs.owner.operand)
        ).res
        rewriter.replace_op(op, ab_bv.FromFixedBitWidthOp(res))


def extend_op_lowering_factory(
    op_cls: type[ab_bv.SignExtendOp] | type[ab_bv.ZeroExtendOp],
    smt_bv_op_cls: type[smt_bv.SignExtendOp] | type[smt_bv.ZeroExtendOp],
) -> type[RewritePattern]:
    @dataclass(frozen=True)
    class ExtendOpLowering(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, op_cls):
                return
            if not isinstance(op.operand.owner, ab_bv.FromFixedBitWidthOp):
                return
            if not isinstance(op.width.owner, ab_bv.ConstantBitWidthOp):
                return
            res = rewriter.insert(
                smt_bv_op_cls(
                    op.operand.owner.operand,
                    smt_bv.BitVectorType(op.width.owner.value.data),
                )
            ).res
            rewriter.replace_op(op, ab_bv.FromFixedBitWidthOp(res))

    return ExtendOpLowering


@dataclass(frozen=True)
class LowerAbbvToBvPass(ModulePass):
    name = "lower-abbv-to-bv"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConstantOpLowering(),
                    FromToFixedBitWidthLowering(),
                    GetBitWidthFromFixedBitWidthLowering(),
                    unary_op_lowering_factory(ab_bv.NegOp, smt_bv.NegOp)(),
                    unary_op_lowering_factory(ab_bv.NotOp, smt_bv.NotOp)(),
                    binary_op_lowering_factory(ab_bv.AddOp, smt_bv.AddOp)(),
                    binary_op_lowering_factory(ab_bv.SubOp, smt_bv.SubOp)(),
                    binary_op_lowering_factory(ab_bv.MulOp, smt_bv.MulOp)(),
                    binary_op_lowering_factory(ab_bv.URemOp, smt_bv.URemOp)(),
                    binary_op_lowering_factory(ab_bv.SRemOp, smt_bv.SRemOp)(),
                    binary_op_lowering_factory(ab_bv.SModOp, smt_bv.SModOp)(),
                    binary_op_lowering_factory(ab_bv.ShlOp, smt_bv.ShlOp)(),
                    binary_op_lowering_factory(ab_bv.LShrOp, smt_bv.LShrOp)(),
                    binary_op_lowering_factory(ab_bv.AShrOp, smt_bv.AShrOp)(),
                    binary_op_lowering_factory(ab_bv.UDivOp, smt_bv.UDivOp)(),
                    binary_op_lowering_factory(ab_bv.SDivOp, smt_bv.SDivOp)(),
                    binary_op_lowering_factory(ab_bv.OrOp, smt_bv.OrOp)(),
                    binary_op_lowering_factory(ab_bv.XorOp, smt_bv.XorOp)(),
                    binary_op_lowering_factory(ab_bv.AndOp, smt_bv.AndOp)(),
                    binary_op_lowering_factory(ab_bv.NAndOp, smt_bv.NAndOp)(),
                    binary_op_lowering_factory(ab_bv.NorOp, smt_bv.NorOp)(),
                    binary_op_lowering_factory(ab_bv.XNorOp, smt_bv.XNorOp)(),
                    CmpOpLowering(),
                    unary_pred_op_lowering_factory(
                        ab_bv.NegOverflowOp, smt_bv.NegOverflowOp
                    )(),
                    ConcatLowering(),
                    extend_op_lowering_factory(
                        ab_bv.SignExtendOp, smt_bv.SignExtendOp
                    )(),
                    extend_op_lowering_factory(
                        ab_bv.ZeroExtendOp, smt_bv.ZeroExtendOp
                    )(),
                ]
            )
        )
        walker.rewrite_module(op)

        DeadCodeElimination().apply(ctx, op)
