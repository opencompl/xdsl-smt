from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import FunctionType
from xdsl.dialects.func import FuncOp, Return

from dialects.smt_dialect import DefineFunOp, ReturnOp
from passes.lower_to_smt.lower_to_smt import LowerToSMT


class ReturnPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Return, rewriter: PatternRewriter):
        if len(op.arguments) != 1:
            raise Exception("Cannot convert functions with multiple results")
        smt_op = ReturnOp(op.arguments[0])
        rewriter.replace_matched_op(smt_op)


class FuncToSMTPattern(RewritePattern):
    """Convert func.func to an SMT formula"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter):
        """
        Convert a `func` function to an smt function.
        """
        # We only handle single-block regions for now
        if len(op.body.blocks) != 1:
            raise Exception("Cannot convert multi-block functions")
        if len(op.function_type.outputs.data) != 1:
            raise Exception("Cannot convert functions with multiple results")

        operand_types = [
            LowerToSMT.lower_type(input) for input in op.function_type.inputs.data
        ]
        result_type = LowerToSMT.lower_type(op.function_type.outputs.data[0])

        # The SMT function replacing the func.func function
        smt_func = DefineFunOp.from_function_type(
            FunctionType.from_lists(operand_types, [result_type]), op.sym_name
        )

        # Replace the old arguments to the new ones
        for i, arg in enumerate(smt_func.body.blocks[0].args):
            op.body.blocks[0].args[i].replace_by(arg)

        # Move the operations to the SMT function
        ops = [op for op in op.body.ops]
        for body_op in ops:
            body_op.detach()
        smt_func.body.blocks[0].add_ops(ops)

        # Replace the arith function with the SMT one
        rewriter.replace_matched_op(smt_func, new_results=[])


func_to_smt_patterns: list[RewritePattern] = [
    FuncToSMTPattern(),
    ReturnPattern(),
]
