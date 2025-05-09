"""
This file is intended to rewrite following operations in SMTBV:

    NegOverflowOp
    UaddOverflowOp
    SaddOverflowOp
    UmulOverflowOp
    SmulOverflowOp

At the current date, May 8, 2025, Z3 doesn't support operations above,
but they are defined in SMT-LIB. So we manually rewrite them by using
existing ones.
"""


from xdsl.dialects.builtin import ModuleOp
from xdsl.context import Context
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.passes import ModulePass
from xdsl_smt.dialects.smt_bitvector_dialect import (
    NegOverflowOp,
    BitVectorType,
    ConstantOp,
    UaddOverflowOp,
    AddOp,
    UltOp,
    UmulOverflowOp,
    UmulNoOverflowOp,
    SmulOverflowOp,
    SaddOverflowOp,
    SmulNoOverflowOp,
    SmulNoUnderflowOp,
    SgtOp,
    SubOp,
    SltOp,
)

from ..dialects.smt_dialect import (
    EqOp,
    OrOp,
    AndOp,
    NotOp,
)
from .dead_code_elimination import DeadCodeElimination


class LowerSaddOverflowOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, saddOverflowOp: SaddOverflowOp, rewriter: PatternRewriter
    ):
        bv_type = saddOverflowOp.lhs.type
        assert isinstance(bv_type, BitVectorType)
        width = bv_type.width.data

        const_zero = ConstantOp.from_int_value(0, width)
        const_min = ConstantOp.from_int_value(1 << (width - 1), width)
        const_max = ConstantOp.from_int_value((1 << (width - 1)) - 1, width)

        const_ops = [const_zero, const_max, const_min]

        lhs = saddOverflowOp.lhs
        rhs = saddOverflowOp.rhs

        # Case1: a>0 && b>0 && (a+b>MAX_SINT) == (a > MAX_SINT - b) -> overflow
        lhs_gt_zero = SgtOp(lhs, const_zero.res)
        rhs_gt_zero = SgtOp(rhs, const_zero.res)
        max_minus_rhs = SubOp(const_max.res, rhs)
        lhs_gt_op = SgtOp(lhs, max_minus_rhs.res)

        case1_and_op1 = AndOp(lhs_gt_zero.res, rhs_gt_zero.res)
        case1_and_op = AndOp(case1_and_op1.res, lhs_gt_op.res)

        case1_ops = [
            lhs_gt_zero,
            rhs_gt_zero,
            max_minus_rhs,
            lhs_gt_op,
            case1_and_op1,
            case1_and_op,
        ]

        # Case2: a<0 && b<0 && (a+b<MIN_SINT) == (a < MIN_SINT - b) -> overflow
        lhs_lt_zero = SltOp(lhs, const_zero.res)
        rhs_lt_zero = SltOp(rhs, const_zero.res)
        min_minus_rhs = SubOp(const_min.res, rhs)
        lhs_lt_op = SltOp(lhs, min_minus_rhs.res)

        case2_and_op1 = AndOp(lhs_lt_zero.res, rhs_lt_zero.res)
        case2_and_op = AndOp(case2_and_op1.res, lhs_lt_op.res)

        case2_ops = [
            lhs_lt_zero,
            rhs_lt_zero,
            min_minus_rhs,
            lhs_lt_op,
            case2_and_op1,
            case2_and_op,
        ]

        last_or_op = OrOp(case1_and_op.res, case2_and_op.res)
        rewriter.replace_op(
            saddOverflowOp, const_ops + case1_ops + case2_ops + [last_or_op]
        )


class LowerSmulOverflowOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, smulOverflow: SmulOverflowOp, rewriter: PatternRewriter
    ):
        # We already have an operation for checking SmulNoOverflowOp and SmulNoUnderflowOp
        # NoOverflow && NoUnderflow -> NoOverflow at all. We just use this condition
        # NoOverflow here might include overflows from both direction

        nooverflow_op = SmulNoOverflowOp(smulOverflow.lhs, smulOverflow.rhs)
        nounderflow_op = SmulNoUnderflowOp(smulOverflow.lhs, smulOverflow.rhs)
        and_op = AndOp(nooverflow_op.res, nounderflow_op.res)
        not_op = NotOp(and_op.res)
        rewriter.replace_op(
            smulOverflow, [nooverflow_op, nounderflow_op, and_op, not_op]
        )
        pass


class LowerUmulOverflowOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, umulOverflow: UmulOverflowOp, rewriter: PatternRewriter
    ):
        # We already have an operation for checking UmulNoOverflow
        nooverflow_op = UmulNoOverflowOp(umulOverflow.lhs, umulOverflow.rhs)
        not_op = NotOp(nooverflow_op.res)
        rewriter.replace_op(umulOverflow, [nooverflow_op, not_op])


class LowerUaddOverflowOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, uaddOverflow: UaddOverflowOp, rewriter: PatternRewriter
    ):
        lhs = uaddOverflow.lhs

        # If the added result is less than any operand, there is an uadd overflow
        uadd_op = AddOp(lhs, uaddOverflow.rhs)
        cmp_op = UltOp(uadd_op.res, lhs)

        rewriter.replace_op(uaddOverflow, [uadd_op, cmp_op])


class LowerNegOverflowOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, negOverflow: NegOverflowOp, rewriter: PatternRewriter):
        operand = negOverflow.operand
        operand_type = operand.type

        assert isinstance(operand_type, BitVectorType)
        width = operand_type.width.data

        # The smallest negative number is 100...00 (in binary).
        const_op = ConstantOp.from_int_value(1 << (width - 1), width)
        eq_op = EqOp(const_op.res, operand)
        rewriter.replace_op(negOverflow, [const_op, eq_op])


class SMTExpand(ModulePass):
    name = "smt-expand"

    def apply(self, ctx: Context, op: ModuleOp):
        # Remove pairs from function arguments.
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerNegOverflowOpPattern(),
                    LowerUaddOverflowOpPattern(),
                    LowerUmulOverflowOpPattern(),
                    LowerSaddOverflowOpPattern(),
                    LowerSmulOverflowOpPattern(),
                ]
            )
        )
        walker.rewrite_module(op)

        # Apply DCE pass
        DeadCodeElimination().apply(ctx, op)
