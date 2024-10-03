from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import PatternRewriter

from xdsl_smt.dialects import smt_ub_dialect
import xdsl_smt.dialects.smt_dialect as smt
import xdsl_smt.dialects.smt_ub_dialect as smt_ub
import xdsl_smt.dialects.smt_utils_dialect as smt_utils
from xdsl_smt.semantics.semantics import EffectStates, RefinementSemantics


class IntegerTypeRefinementSemantics(RefinementSemantics):
    def get_semantics(
        self,
        val_before: SSAValue,
        val_after: SSAValue,
        states_before: EffectStates,
        states_after: EffectStates,
        rewriter: PatternRewriter,
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
        refinement_integer = smt.ImpliesOp(not_before_poison.res, not_poison_eq.res)
        rewriter.insert_op_before_matched_op(
            [
                not_before_poison,
                not_after_poison,
                eq_vals,
                not_poison_eq,
                refinement_integer,
            ]
        )

        if len(states_before.states) == 0:
            return refinement_integer.res

        if len(states_before.states) != 1:
            raise ValueError("Integer refinement can only handle UB state for now")

        # With UB, our refinement is: ub_before \/ (not ub_after /\ integer_refinement)
        ub_before = states_before.states[smt_ub.UBStateType()]
        ub_before_bool = smt_ub.ToBoolOp(ub_before)
        ub_after = states_after.states[smt_ub.UBStateType()]
        ub_after_bool = smt_ub.ToBoolOp(ub_after)
        not_ub_after = smt.NotOp(ub_after_bool.res)
        not_ub_before_case = smt.AndOp(not_ub_after.res, refinement_integer.res)
        refinement = smt.OrOp(ub_before_bool.res, not_ub_before_case.res)
        rewriter.insert_op_before_matched_op(
            [
                ub_before_bool,
                ub_after_bool,
                not_ub_after,
                not_ub_before_case,
                refinement,
            ]
        )
        return refinement.res
