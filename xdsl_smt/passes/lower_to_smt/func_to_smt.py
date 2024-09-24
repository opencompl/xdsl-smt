from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.dialects.builtin import FunctionType
from xdsl.dialects.func import FuncOp, Return

from xdsl_smt.passes.lower_to_smt import (
    SMTLowerer,
    SMTLoweringRewritePattern,
)
from xdsl_smt.semantics.semantics import EffectStates
from xdsl_smt.utils.rewrite_tools import new_ops
from xdsl_smt.dialects.smt_dialect import DefineFunOp, ReturnOp
from xdsl_smt.dialects.smt_utils_dialect import pair_from_list as smt_pair_from_list


class ReturnPattern(SMTLoweringRewritePattern):
    def rewrite(
        self,
        op: Operation,
        effect_states: EffectStates,
        rewriter: PatternRewriter,
        smt_lowerer: SMTLowerer,
    ) -> EffectStates:
        assert isinstance(op, Return)
        smt_op = ReturnOp(
            smt_pair_from_list(
                *op.arguments,
                *(EffectStates.states[effect] for effect in SMTLowerer.effect_types)
            )
        )
        rewriter.replace_matched_op([*new_ops(smt_op)])
        return effect_states

    def match_and_rewrite(self, op: Return, rewriter: PatternRewriter):
        smt_op = ReturnOp(smt_pair_from_list(*op.arguments))
        rewriter.replace_matched_op([*new_ops(smt_op)])


class FuncToSMTPattern(SMTLoweringRewritePattern):
    """
    Convert a `func.func` to its SMT semantics.
    `func.func` semantics is an SMT function, with one new argument per effect kind
    we handle.
    """

    def rewrite(
        self,
        op: Operation,
        effect_states: EffectStates,
        rewriter: PatternRewriter,
        smt_lowerer: SMTLowerer,
    ) -> EffectStates:
        assert isinstance(op, FuncOp)

        # Lower the function body
        result_values, new_states = smt_lowerer.lower_region(op.body, effect_states)

        """Convert a list of types into a cons-list of SMT pairs"""

        # Get the operands and result types
        operand_types = [
            smt_lowerer.lower_type(input) for input in op.function_type.inputs.data
        ]
        result_types = (
            [smt_pair_from_list(*result_values).type] if result_values else []
        )

        # Create the new SMT function
        region = op.detach_region(op.body)
        smt_func = DefineFunOp.build(
            result_types=[FunctionType.from_lists(operand_types, result_types)],
            attributes={"fun_name": op.sym_name},
            regions=[region],
        )
        rewriter.replace_matched_op(smt_func, new_results=[])

        return new_states


func_to_smt_patterns: dict[type[Operation], SMTLoweringRewritePattern] = {
    FuncOp: FuncToSMTPattern(),
    Return: ReturnPattern(),
}
