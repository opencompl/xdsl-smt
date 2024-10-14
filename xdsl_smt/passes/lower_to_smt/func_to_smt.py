from xdsl.ir import Operation, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.dialects.builtin import FunctionType
from xdsl.dialects.func import FuncOp, Return

from xdsl_smt.dialects.effects.effect import StateType
from xdsl_smt.passes.lower_to_smt import (
    SMTLowerer,
    SMTLoweringRewritePattern,
)
from xdsl_smt.dialects.smt_dialect import DefineFunOp, ReturnOp
from xdsl_smt.dialects.smt_utils_dialect import merge_values_with_pairs
from xdsl_smt.dialects.smt_utils_dialect import (
    pair_type_from_list as smt_pair_type_from_list,
)


class ReturnPattern(SMTLoweringRewritePattern):
    def rewrite(
        self,
        op: Operation,
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
        smt_lowerer: SMTLowerer,
    ) -> SSAValue | None:
        assert isinstance(op, Return)
        assert effect_state is not None
        result = merge_values_with_pairs(
            (*op.arguments, effect_state),
            rewriter,
            InsertPoint.before(op),
        )
        assert result is not None, "Cannot handle return with no arguments"

        smt_op = ReturnOp(result)
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
        assert isinstance(op, FuncOp)

        # Get the operands and result types
        # It is the original operansd and result types, lowered to SMT, plus the
        # effect states.
        operand_types = [
            smt_lowerer.lower_type(input) for input in op.function_type.inputs.data
        ] + [StateType()]
        result_types = [
            smt_lowerer.lower_type(result) for result in op.function_type.outputs.data
        ] + [StateType()]
        result_type = smt_pair_type_from_list(*result_types)

        # Create the new SMT function
        region = op.detach_region(op.body)
        smt_func = DefineFunOp.build(
            result_types=[FunctionType.from_lists(operand_types, [result_type])],
            attributes={"fun_name": op.sym_name},
            regions=[region],
        )
        rewriter.replace_matched_op(smt_func, new_results=[])

        # Append a block argument for the effect state
        region_state = region.block.insert_arg(StateType(), len(region.block.args))

        # Lower the function body
        smt_lowerer.lower_region(region, region_state)

        return effect_state


func_to_smt_patterns: dict[type[Operation], SMTLoweringRewritePattern] = {
    FuncOp: FuncToSMTPattern(),
    Return: ReturnPattern(),
}
