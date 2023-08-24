from xdsl.irdl import IRDLOperation
from xdsl.pattern_rewriter import RewritePattern, PatternRewriter
from xdsl.ir import Operation

from dialects import smt_bitvector_dialect as smt_bv
from dialects import transfer


def trivial_pattern(
    match_type: type[IRDLOperation], rewrite_type: type[IRDLOperation]
) -> RewritePattern:
    class TrivialBinOpPattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, match_type):
                return
            # TODO: How to handle multiple results, or results with different types?
            new_op = rewrite_type.create(
                operands=op.operands,
                result_types=[op.operands[0].type],
            )
            rewriter.replace_matched_op([new_op])

    return TrivialBinOpPattern()


transfer_to_smt_patterns: list[RewritePattern] = [
    trivial_pattern(transfer.AndOp, smt_bv.AndOp),
    trivial_pattern(transfer.OrOp, smt_bv.OrOp),
]
