from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import PatternRewriter

import xdsl_smt.dialects.smt_dialect as smt
import xdsl_smt.dialects.smt_utils_dialect as smt_utils
from xdsl_smt.semantics.semantics import RefinementSemantics


class IntegerTypeRefinementSemantics(RefinementSemantics):
    def get_semantics(
        self, val_before: SSAValue, val_after: SSAValue, rewriter: PatternRewriter
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
