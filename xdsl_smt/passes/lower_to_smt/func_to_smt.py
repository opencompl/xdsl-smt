"""
Define the lowering rewrite patterns to convert the func dialect to the semantics
dialects.
"""

from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.dialects import func

from xdsl_smt.dialects.effects.effect import StateType
from xdsl_smt.passes.lower_to_smt.smt_lowerer import (
    SMTLowerer,
)
from xdsl_smt.passes.lower_to_smt.smt_rewrite_patterns import SMTLoweringRewritePattern
from xdsl_smt.dialects.smt_dialect import DefineFunOp, ReturnOp


class ReturnPattern(SMTLoweringRewritePattern):
    """Lower a `func.return` to an SMT return operation."""

    def rewrite(
        self,
        op: Operation,
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
        smt_lowerer: SMTLowerer,
    ) -> SSAValue | None:
        assert isinstance(op, func.ReturnOp)
        assert effect_state is not None
        smt_op = ReturnOp([*op.arguments, effect_state])
        rewriter.replace_matched_op([smt_op])
        return effect_state


class FuncToSMTPattern(SMTLoweringRewritePattern):
    """
    Convert a `func.func` to its SMT semantics.
    `func.func` semantics is an SMT function, with one new argument per effect kind
    we handle.
    """

    def rewrite(
        self,
        op: Operation,
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
        smt_lowerer: SMTLowerer,
    ) -> SSAValue | None:
        """Convert a list of types into a cons-list of SMT pairs"""
        assert isinstance(op, func.FuncOp)

        region = op.detach_region(op.body)
        # Append a block argument for the effect state
        region_state = region.block.insert_arg(StateType(), len(region.block.args))

        # Lower the function body
        smt_lowerer.lower_region(region, region_state)

        # Create the new SMT function
        smt_func = DefineFunOp(region, op.sym_name)
        rewriter.replace_matched_op(smt_func, new_results=[])

        # The effect state is unchanged
        return effect_state


func_to_smt_patterns: dict[type[Operation], SMTLoweringRewritePattern] = {
    func.FuncOp: FuncToSMTPattern(),
    func.ReturnOp: ReturnPattern(),
}
"""The set of patterns to lower the func dialect to the semantics dialect."""
