from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.hints import isa

import xdsl_smt.dialects.smt_dialect as smt
import xdsl_smt.dialects.smt_utils_dialect as smt_utils


def nonpoison_to_poison_refinemnet(
    val_before: SSAValue, val_after: SSAValue, rewriter: PatternRewriter
) -> SSAValue:
    """Compute the refinement from a value with no poison semantics to a value with poison semantics."""
    after_val = smt_utils.FirstOp(val_after)
    after_poison = smt_utils.SecondOp(val_after)

    rewriter.insert_op_before_matched_op([after_val, after_poison])

    eq_vals = smt.EqOp(val_before, after_val.res)
    not_poison_eq = smt.NotOp(after_poison.res)
    refinement = smt.AndOp(eq_vals.res, not_poison_eq.res)
    rewriter.insert_op_before_matched_op([eq_vals, not_poison_eq, refinement])
    return refinement.res


def poison_to_nonpoison_refinement(
    val_before: SSAValue, val_after: SSAValue, rewriter: PatternRewriter
) -> SSAValue:
    """Compute the refinement from a value with poison semantics to a value with no poison semantics."""
    before_poison = smt_utils.SecondOp(val_before)
    before_val = smt_utils.FirstOp(val_before)

    rewriter.insert_op_before_matched_op([before_poison, before_val])

    eq_vals = smt.EqOp(before_val.res, val_after)
    not_poison_eq = smt.NotOp(before_poison.res)
    refinement = smt.ImpliesOp(not_poison_eq.res, eq_vals.res)
    rewriter.insert_op_before_matched_op([eq_vals, not_poison_eq, refinement])
    return refinement.res


def poison_to_poison_refinement(
    val_before: SSAValue, val_after: SSAValue, rewriter: PatternRewriter
) -> SSAValue:
    """Compute the refinement from a value with poison semantics to a value with poison semantics."""
    before_poison = smt_utils.SecondOp(val_before)
    after_poison = smt_utils.SecondOp(val_after)

    before_val = smt_utils.FirstOp(val_before)
    after_val = smt_utils.FirstOp(val_after)

    rewriter.insert_op_before_matched_op(
        [
            before_poison,
            after_poison,
            before_val,
            after_val,
        ]
    )

    not_before_poison = smt.NotOp(before_poison.res)
    not_after_poison = smt.NotOp(after_poison.res)
    eq_vals = smt.EqOp(before_val.res, after_val.res)
    not_poison_eq = smt.AndOp(eq_vals.res, not_after_poison.res)
    refinement = smt.ImpliesOp(not_before_poison.res, not_poison_eq.res)
    rewriter.insert_op_before_matched_op(
        [
            not_before_poison,
            not_after_poison,
            eq_vals,
            not_poison_eq,
            refinement,
        ]
    )
    return refinement.res


def nonpoison_to_nonpoison_refinement(
    val_before: SSAValue, val_after: SSAValue, rewriter: PatternRewriter
) -> SSAValue:
    """Compute the refinement from a value with no poison semantics to a value with no poison semantics."""
    eq_op = smt.EqOp(val_before, val_after)
    rewriter.insert_op_before_matched_op(eq_op)
    return eq_op.res


def optionally_poison_refinement(
    val_before: SSAValue, val_after: SSAValue, rewriter: PatternRewriter
) -> SSAValue:
    """
    Compute the refinement from two values that may or may not have poison semantics.
    This is decided by the value types.
    """
    if isa(val_before.type, smt_utils.AnyPairType):
        if isa(val_after.type, smt_utils.AnyPairType):
            return poison_to_poison_refinement(val_before, val_after, rewriter)
        else:
            return poison_to_nonpoison_refinement(val_before, val_after, rewriter)
    else:
        if isa(val_after.type, smt_utils.AnyPairType):
            return nonpoison_to_poison_refinemnet(val_before, val_after, rewriter)
        else:
            return nonpoison_to_nonpoison_refinement(val_before, val_after, rewriter)
