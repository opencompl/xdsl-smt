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
from ...dialects.smt_utils_dialect import PairType, SecondOp, FirstOp, PairOp
from xdsl_smt.dialects.smt_dialect import BoolType
from ...utils.transfer_to_smt_util import get_low_bits, set_high_bits, count_lzeros, count_rzeros, count_lones, \
    count_rones


def abstract_value_type_lowerer(type: Attribute) -> PairType | None:
    """Lower all types in an abstract value to SMT types
    But the last element is useless, this makes GetOp easier"""
    if isinstance(type, transfer.AbstractValueType):
        result = PairType(LowerToSMT.lower_type(type.get_fields()[-1]), BoolType())
        for ty in reversed(type.get_fields()[:-1]):
            result = PairType(LowerToSMT.lower_type(ty), result)
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
    new_ops = [smt.EqOp,
               smt.EqOp,
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

        rewrite_type = self.new_ops[predicate]
        new_op = rewrite_type.create(
            operands=op.operands,
            result_types=[BoolType()]
        )

        resList = [new_op]

        # Neq -> Not(Eq....)
        if 1 == predicate:
            tmp_op = smt.NotOp(new_op.results[0])
            resList.append(tmp_op)

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(resList[-1].results[0], b1.results[0], b0.results[0])

        poison_op = smt.ConstantBoolOp.from_bool(False)
        res_op = PairOp(bool_to_bv.results[0], poison_op.res)

        resList += [b1, b0, bool_to_bv, poison_op, res_op]
        rewriter.replace_matched_op(resList)
        return resList


class GetOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.GetOp, rewriter: PatternRewriter):
        index = op.attributes["index"].value.data
        arg = op.operands[0]
        insertOps = []
        while index != 0:
            insertOps.append(SecondOp(arg))
            arg = insertOps[-1].results[0]
            rewriter.insert_op_before_matched_op(insertOps[-1])
            index -= 1

        new_op = FirstOp(arg)
        rewriter.replace_matched_op([new_op])


class MakeOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.MakeOp, rewriter: PatternRewriter):
        argList = op.arguments
        # The last element is useless, getOp won't access it
        # So it can be any value
        opList = [PairOp(argList[-1], argList[-1])]
        result = opList[-1].results[0]
        for ty in reversed(argList[:-1]):
            opList.append(PairOp(ty, result))
            result = opList[-1].results[0]
        for op in opList[:-1]:
            rewriter.insert_op_before_matched_op(op)
        rewriter.replace_matched_op(opList[-1])


class GetBitWidthOpPattern(RewritePattern):
    '''1. There is no direct API to obtain the bit width of a vector.
    2. All bit width are fixed values.
    As a result, we replace this method with a constant.'''

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.GetBitWidthOp, rewriter: PatternRewriter):
        width = op.operands[0].type.width
        value = width
        rewriter.replace_matched_op(smt_bv.ConstantOp(value, width))


class ConstantOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.Constant, rewriter: PatternRewriter):
        width = op.operands[0].type.width
        value = op.attributes["value"].value.data
        rewriter.replace_matched_op(smt_bv.ConstantOp(value, width))


class CountLOneOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.CountLOneOp, rewriter: PatternRewriter):
        operand = op.operands[0]
        resList = count_lones(operand)
        for op in resList[:-1]:
            rewriter.insert_op_before_matched_op(op)
        rewriter.replace_matched_op(resList[-1])


class CountLZeroOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.CountLZeroOp, rewriter: PatternRewriter):
        operand = op.operands[0]
        resList = count_lzeros(operand)
        for op in resList[:-1]:
            rewriter.insert_op_before_matched_op(op)
        rewriter.replace_matched_op(resList[-1])


class CountROneOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.CountROneOp, rewriter: PatternRewriter):
        operand = op.operands[0]
        resList = count_rones(operand)
        for op in resList[:-1]:
            rewriter.insert_op_before_matched_op(op)
        rewriter.replace_matched_op(resList[-1])


class CountRZeroOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.CountRZeroOp, rewriter: PatternRewriter):
        operand = op.operands[0]
        resList = count_rzeros(operand)
        for op in resList[:-1]:
            rewriter.insert_op_before_matched_op(op)
        rewriter.replace_matched_op(resList[-1])


class SetHighBitsOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.SetHighBitsOp, rewriter: PatternRewriter):
        result = set_high_bits(op.operands[0], op.operands[1])
        for op in result[:-1]:
            rewriter.insert_op_before_matched_op(op)
        rewriter.replace_matched_op(result[-1])


class GetLowBitsOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.GetLowBitsOp, rewriter: PatternRewriter):
        result = get_low_bits(op.operands[0], op.operands[1])
        for op in result[:-1]:
            rewriter.insert_op_before_matched_op(op)
        rewriter.replace_matched_op(result[-1])


class SMinOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.SMinOp, rewriter: PatternRewriter):
        smin_if = smt_bv.UleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        rewriter.replace_matched_op(smt.IteOp(smin_if.results[0], op.operands[0], op.operands[1]))


class SMaxOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.SMaxOp, rewriter: PatternRewriter):
        smax_if = smt_bv.SleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(smax_if)
        rewriter.replace_matched_op(smt.IteOp(smax_if.results[0], op.operands[1], op.operands[0]))

    pass


class UMaxOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.UMaxOp, rewriter: PatternRewriter):
        umax_if = smt_bv.UleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(umax_if)
        rewriter.replace_matched_op(smt.IteOp(umax_if.results[0], op.operands[1], op.operands[0]))


class UMinOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.UMinOp, rewriter: PatternRewriter):
        umin_if = smt_bv.UleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(umin_if)
        rewriter.replace_matched_op(smt.IteOp(umin_if.results[0], op.operands[0], op.operands[1]))


transfer_to_smt_patterns: list[RewritePattern] = [
    trivial_pattern(transfer.AndOp, smt_bv.AndOp),
    trivial_pattern(transfer.OrOp, smt_bv.OrOp),
    trivial_pattern(transfer.SubOp, smt_bv.SubOp),
    trivial_pattern(transfer.AddOp, smt_bv.AddOp),
    trivial_pattern(transfer.MulOp, smt_bv.MulOp),
    trivial_pattern(transfer.UMulOverflowOp, smt_bv.UmulNoOverflowOp),
    trivial_pattern(transfer.NegOp, smt_bv.NegOp),
    CmpOpPattern(),
    GetOpPattern(),
    MakeOpPattern(),
    ConstantOpPattern(),
    GetBitWidthOpPattern(),
    CountLOneOpPattern(),
    CountLZeroOpPattern(),
    CountROneOpPattern(),
    CountRZeroOpPattern(),
    SetHighBitsOpPattern(),
    GetLowBitsOpPattern(),
    SMinOpPattern(),
    SMaxOpPattern(),
    UMaxOpPattern(),
    UMinOpPattern(),
]
