from xdsl.irdl import IRDLOperation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.ir import Operation

from ...dialects import smt_bitvector_dialect as smt_bv
from ...dialects import smt_dialect as smt
from ...dialects import transfer
from xdsl.ir import Attribute, MLContext
from .lower_to_smt import LowerToSMT
from ...dialects.smt_utils_dialect import PairType
from xdsl_smt.dialects.smt_dialect import BoolType

def abstract_value_type_lowerer(type: Attribute) -> PairType | None:
    """Lower all types in an abstract value to SMT types"""
    if isinstance(type, transfer.AbstractValueType):
        result=PairType(LowerToSMT.lower_type(type.get_fields()[-1]), BoolType())
        for ty in reversed(type.get_fields()[:-1]):
            result=PairType(LowerToSMT.lower_type(ty),result)
        return result
    return None


def transfer_integer_type_lowerer(type: Attribute, width: int) -> smt_bv.BitVectorType | None:
    if isinstance(type, transfer.TransIntegerType):
        return smt_bv.BitVectorType(width)
    return None

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


class CmpOpPattern(RewritePattern):
    new_ops=[ smt.EqOp,
              smt.NotOp,
              smt_bv.SltOp,
              smt_bv.SleOp,
              smt_bv.SgtOp,
              smt_bv.SgeOp,
              smt_bv.UltOp,
              smt_bv.UleOp,
              smt_bv.UgtOp,
              smt_bv.UgeOp,
    ]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.CmpOp, rewriter: PatternRewriter):
        predicate = op.attributes["predicate"].value.data
        #Neq -> Not(Eq....)
        if 1==predicate:
            tmp_op=smt.EqOp.create(
                operands=op.operands,
                result_types=[op.operands[0].type],
            )
            rewriter.insert_op_before_matched_op(tmp_op)
            op=tmp_op

        rewrite_type = self.new_ops[predicate]
        new_op = rewrite_type.create(
            operands=op.results,
            result_types=[op.operands[0].type],
        )
        rewriter.replace_matched_op([new_op])


transfer_to_smt_patterns: list[RewritePattern] = [
    trivial_pattern(transfer.AndOp, smt_bv.AndOp),
    trivial_pattern(transfer.OrOp, smt_bv.OrOp),
    trivial_pattern(transfer.SubOp, smt_bv.SubOp),
    trivial_pattern(transfer.AddOp, smt_bv.AddOp),
    trivial_pattern(transfer.MulOp, smt_bv.MulOp),
    trivial_pattern(transfer.UMulOverflowOp, smt_bv.UmulNoOverflowOp),
    CmpOpPattern()
]
