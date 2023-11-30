"""
This file contains semantics of constraints and rewrites for integer arithmetic
in PDL.
"""

from xdsl.ir import Operation
from xdsl.pattern_rewriter import PatternRewriter

from xdsl.dialects.pdl import ApplyNativeRewriteOp
import xdsl_smt.dialects.smt_bitvector_dialect as smt_bv


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


integer_arith_native_rewrites = {
    "addi": addi_rewrite,
    "subi": subi_rewrite,
    "muli": muli_rewrite,
}
