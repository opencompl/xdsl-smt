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
    SgeOp,
)

from xdsl_smt.dialects.smt_dialect import (
    EqOp,
    AndOp,
    NotOp,
    DistinctOp,
)
from .dead_code_elimination import DeadCodeElimination


class LowerSaddOverflowOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, sadd_overflow_op: SaddOverflowOp, rewriter: PatternRewriter
    ):
        bv_type = sadd_overflow_op.lhs.type
        assert isinstance(bv_type, BitVectorType)
        width = bv_type.width.data

        lhs = sadd_overflow_op.lhs
        rhs = sadd_overflow_op.rhs
        add_op = AddOp(lhs, rhs)

        const_zero_op = ConstantOp(0, width)

        # Here we use LLVM's implementation:
        # Overflow = isNonNegative() == RHS.isNonNegative() &&
        #              Res.isNonNegative() != isNonNegative();
        lhs_non_negative_op = SgeOp(lhs, const_zero_op.res)
        rhs_non_negative_op = SgeOp(rhs, const_zero_op.res)
        result_non_negative_op = SgeOp(add_op.res, const_zero_op.res)

        lhs_sign_eq_rhs_sign_op = EqOp(lhs_non_negative_op.res, rhs_non_negative_op.res)
        result_sign_ne_lhs_sign_op = DistinctOp(
            lhs_sign_eq_rhs_sign_op.res, result_non_negative_op.res
        )
        overflow_op = AndOp(lhs_sign_eq_rhs_sign_op.res, result_sign_ne_lhs_sign_op.res)
        rewriter.replace_op(
            sadd_overflow_op,
            [
                const_zero_op,
                add_op,
                lhs_non_negative_op,
                rhs_non_negative_op,
                result_non_negative_op,
                lhs_sign_eq_rhs_sign_op,
                result_sign_ne_lhs_sign_op,
                overflow_op,
            ],
        )


class LowerSmulOverflowOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, smul_overflow_op: SmulOverflowOp, rewriter: PatternRewriter
    ):
        # We already have an operation for checking SmulNoOverflowOp and SmulNoUnderflowOp
        # NoOverflow && NoUnderflow -> NoOverflow at all. We just use this condition
        # NoOverflow here might include overflows from both direction

        nooverflow_op = SmulNoOverflowOp(smul_overflow_op.lhs, smul_overflow_op.rhs)
        nounderflow_op = SmulNoUnderflowOp(smul_overflow_op.lhs, smul_overflow_op.rhs)
        and_op = AndOp(nooverflow_op.res, nounderflow_op.res)
        not_op = NotOp(and_op.res)
        rewriter.replace_op(
            smul_overflow_op, [nooverflow_op, nounderflow_op, and_op, not_op]
        )


class LowerUmulOverflowOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, umul_overflow_op: UmulOverflowOp, rewriter: PatternRewriter
    ):
        # We already have an operation for checking UmulNoOverflow
        nooverflow_op = UmulNoOverflowOp(umul_overflow_op.lhs, umul_overflow_op.rhs)
        not_op = NotOp(nooverflow_op.res)
        rewriter.replace_op(umul_overflow_op, [nooverflow_op, not_op])


class LowerUaddOverflowOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, uadd_overflow_op: UaddOverflowOp, rewriter: PatternRewriter
    ):
        lhs = uadd_overflow_op.lhs

        # If the added result is less than any operand, there is an uadd overflow
        uadd_op = AddOp(lhs, uadd_overflow_op.rhs)
        cmp_op = UltOp(uadd_op.res, lhs)

        rewriter.replace_op(uadd_overflow_op, [uadd_op, cmp_op])


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
