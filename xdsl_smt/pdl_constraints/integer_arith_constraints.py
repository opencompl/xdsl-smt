"""
This file contains semantics of constraints and rewrites for integer arithmetic
in PDL.
"""

from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter

from xdsl.dialects.pdl import ApplyNativeConstraintOp, ApplyNativeRewriteOp
import xdsl_smt.dialects.smt_bitvector_dialect as smt_bv
import xdsl_smt.dialects.smt_dialect as smt


def single_op_rewrite(
    op: ApplyNativeRewriteOp, rewriter: PatternRewriter, op_type: type[Operation]
):
    lhs, rhs = op.args
    new_op = op_type.create(operands=[lhs, rhs], result_types=[lhs.type])
    rewriter.replace_matched_op(new_op)


def addi_rewrite(op: ApplyNativeRewriteOp, rewriter: PatternRewriter):
    return single_op_rewrite(op, rewriter, smt_bv.AddOp)


def subi_rewrite(op: ApplyNativeRewriteOp, rewriter: PatternRewriter):
    return single_op_rewrite(op, rewriter, smt_bv.SubOp)


def muli_rewrite(op: ApplyNativeRewriteOp, rewriter: PatternRewriter):
    return single_op_rewrite(op, rewriter, smt_bv.MulOp)


def is_minus_one(op: ApplyNativeConstraintOp, rewriter: PatternRewriter) -> SSAValue:
    (value,) = op.args

    if not isinstance(value.type, smt_bv.BitVectorType):
        raise Exception(
            "is_minus_one expects the input to be lowered to a `!smt.bv<...>`"
        )

    width = value.type.width.data
    minus_one = smt_bv.ConstantOp(2**width - 1, width)
    eq_minus_one = smt.EqOp(value, minus_one.res)
    rewriter.replace_matched_op([eq_minus_one, minus_one], [])
    return eq_minus_one.res


integer_arith_native_rewrites = {
    "addi": addi_rewrite,
    "subi": subi_rewrite,
    "muli": muli_rewrite,
}

integer_arith_native_constraints = {
    "is_minus_one": is_minus_one,
}
