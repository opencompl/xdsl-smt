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
from xdsl.ir import Attribute
from .lower_to_smt import LowerToSMT
from xdsl.dialects.func import Call
from ...dialects.smt_utils_dialect import PairType, SecondOp, FirstOp, PairOp
from xdsl_smt.dialects.smt_dialect import BoolType, DefineFunOp, CallOp
from xdsl.dialects.builtin import ModuleOp
from ...utils.transfer_to_smt_util import (
    get_low_bits,
    set_high_bits,
    count_lzeros,
    count_rzeros,
    count_lones,
    count_rones,
)


def abstract_value_type_lowerer(
    type: Attribute,
) -> PairType[Attribute, Attribute] | None:
    """Lower all types in an abstract value to SMT types
    But the last element is useless, this makes GetOp easier"""
    if isinstance(type, transfer.AbstractValueType):
        result = PairType(LowerToSMT.lower_type(type.get_fields()[-1]), BoolType())
        for ty in reversed(type.get_fields()[:-1]):
            result = PairType(LowerToSMT.lower_type(ty), result)
        return result
    return None


def transfer_integer_type_lowerer(
    type: Attribute, width: int
) -> smt_bv.BitVectorType | None:
    if isinstance(type, transfer.TransIntegerType):
        return smt_bv.BitVectorType(width)
    return None


def trivial_pattern(
    match_type: type[IRDLOperation], rewrite_type: type[IRDLOperation]
) -> RewritePattern:
    class TrivialBinOpPattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
            if not isinstance(op, match_type):
                return
            # TODO: How to handle multiple results, or results with different types?
            new_op = rewrite_type.create(
                operands=op.operands,
                result_types=[op.operands[0].type],
            )
            rewriter.replace_matched_op([new_op])

    return TrivialBinOpPattern()


class UMulOverflowOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.UMulOverflowOp, rewriter: PatternRewriter) -> None:
        suml_nooverflow=smt_bv.UmulNoOverflowOp.get(op.operands[0], op.operands[1])

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(suml_nooverflow.res, b1.res, b0.res)
        poison_op = smt.ConstantBoolOp.from_bool(False)
        res=PairOp(bool_to_bv.res, poison_op.res)
        rewriter.replace_matched_op([suml_nooverflow,b0,b1,bool_to_bv,poison_op,res])


class CmpOpPattern(RewritePattern):
    new_ops = [
        smt.EqOp,
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
    def match_and_rewrite(self, op: transfer.CmpOp, rewriter: PatternRewriter) -> None:
        predicate = op.predicate.value.data

        rewrite_type = self.new_ops[predicate]
        new_op = rewrite_type.create(operands=op.operands, result_types=[BoolType()])

        resList: list[Operation] = [new_op]

        # Neq -> Not(Eq....)
        if 1 == predicate:
            tmp_op = smt.NotOp.get(new_op.results[0])
            resList.append(tmp_op)

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(resList[-1].results[0], b1.results[0], b0.results[0])

        poison_op = smt.ConstantBoolOp.from_bool(False)
        res_op = PairOp(bool_to_bv.results[0], poison_op.res)

        resList += [b1, b0, bool_to_bv, poison_op, res_op]
        rewriter.replace_matched_op(resList)
        # return resList


class GetOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.GetOp, rewriter: PatternRewriter) -> None:
        index = op.index.value.data
        arg = op.operands[0]
        insertOps: list[Operation] = []
        while index != 0:
            insertOps.append(SecondOp(arg))
            arg = insertOps[-1].results[0]
            rewriter.insert_op_before_matched_op(insertOps[-1])
            index -= 1

        new_op = FirstOp(arg)
        rewriter.replace_matched_op([new_op])


class MakeOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.MakeOp, rewriter: PatternRewriter) -> None:
        argList = op.arguments
        # The last element is useless, getOp won't access it
        # So it can be any bool value
        false_constant = smt.ConstantBoolOp.from_bool(False)
        opList: list[Operation] = [PairOp(argList[-1], false_constant.res)]
        result = opList[-1].results[0]
        for ty in reversed(argList[:-1]):
            opList.append(PairOp(ty, result))
            result = opList[-1].results[0]

        rewriter.insert_op_before_matched_op(false_constant)
        for newOp in opList[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.replace_matched_op(opList[-1])


class GetBitWidthOpPattern(RewritePattern):
    """1. There is no direct API to obtain the bit width of a vector.
    2. All bit width are fixed values.
    As a result, we replace this method with a constant."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: transfer.GetBitWidthOp, rewriter: PatternRewriter
    ) -> None:
        assert isinstance(op.op.type, smt_bv.BitVectorType)
        width = op.op.type.width
        value = width
        rewriter.replace_matched_op(smt_bv.ConstantOp(value, width))


class ConstantOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: transfer.Constant, rewriter: PatternRewriter
    ) -> None:
        assert isinstance(op.op.type, smt_bv.BitVectorType)
        width = op.op.type.width
        value = op.value.value.data
        rewriter.replace_matched_op(smt_bv.ConstantOp(value, width))


class CountLOneOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: transfer.CountLOneOp, rewriter: PatternRewriter
    ) -> None:
        operand = op.operands[0]
        resList = count_lones(operand)
        for newOp in resList[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.replace_matched_op(resList[-1])


class CountLZeroOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: transfer.CountLZeroOp, rewriter: PatternRewriter
    ) -> None:
        operand = op.operands[0]
        resList = count_lzeros(operand)
        for newOp in resList[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.replace_matched_op(resList[-1])


class CountROneOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: transfer.CountROneOp, rewriter: PatternRewriter
    ) -> None:
        operand = op.operands[0]
        resList, afterList = count_rones(operand)
        for newOp in resList[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        for newOp in afterList:
            rewriter.insert_op_after_matched_op(newOp)
        rewriter.replace_matched_op(resList[-1])


class CountRZeroOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: transfer.CountRZeroOp, rewriter: PatternRewriter
    ) -> None:
        operand = op.operands[0]
        resList, afterList = count_rzeros(operand)
        for newOp in resList[:-1]:
            rewriter.insert_op_after_matched_op(newOp)
        for newOp in afterList:
            rewriter.insert_op_after_matched_op(newOp)
        #countRzero is different, it insert operation after the matched op.
        rewriter.replace_matched_op(resList[-1])


class SetHighBitsOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: transfer.SetHighBitsOp, rewriter: PatternRewriter
    ) -> None:
        result = set_high_bits(op.operands[0], op.operands[1])
        for newOp in result[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.replace_matched_op(result[-1])


class GetLowBitsOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: transfer.GetLowBitsOp, rewriter: PatternRewriter
    ) -> None:
        result = get_low_bits(op.operands[0], op.operands[1])
        for newOp in result[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.replace_matched_op(result[-1])


class SMinOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.SMinOp, rewriter: PatternRewriter) -> None:
        smin_if = smt_bv.UleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        rewriter.replace_matched_op(
            smt.IteOp(smin_if.results[0], op.operands[0], op.operands[1])
        )


class SMaxOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.SMaxOp, rewriter: PatternRewriter) -> None:
        smax_if = smt_bv.SleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(smax_if)
        rewriter.replace_matched_op(
            smt.IteOp(smax_if.results[0], op.operands[1], op.operands[0])
        )



class UMaxOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.UMaxOp, rewriter: PatternRewriter) -> None:
        umax_if = smt_bv.UleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(umax_if)
        rewriter.replace_matched_op(
            smt.IteOp(umax_if.results[0], op.operands[1], op.operands[0])
        )


class UMinOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: transfer.UMinOp, rewriter: PatternRewriter) -> None:
        umin_if = smt_bv.UleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(umin_if)
        rewriter.replace_matched_op(
            smt.IteOp(umin_if.results[0], op.operands[0], op.operands[1])
        )

class CallOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Call, rewriter: PatternRewriter) -> None:
        moduleOp=op.parent_op()
        callee=op.callee.string_value()
        while not isinstance(moduleOp, ModuleOp):
            moduleOp=moduleOp.parent_op()
        isinstance(moduleOp, ModuleOp)
        for funcOp in moduleOp.ops:
            if isinstance(funcOp, DefineFunOp):
                if funcOp.fun_name.data==callee:
                    newCallOp=CallOp.get(funcOp.results[0], op.arguments)
                    rewriter.replace_matched_op(newCallOp)
                    return
        assert False and "Cannot find the desired call"


transfer_to_smt_patterns: list[RewritePattern] = [
    trivial_pattern(transfer.AndOp, smt_bv.AndOp),
    trivial_pattern(transfer.OrOp, smt_bv.OrOp),
    trivial_pattern(transfer.XorOp, smt_bv.XorOp),
    trivial_pattern(transfer.SubOp, smt_bv.SubOp),
    trivial_pattern(transfer.AddOp, smt_bv.AddOp),
    trivial_pattern(transfer.MulOp, smt_bv.MulOp),
    trivial_pattern(transfer.NegOp, smt_bv.NotOp),
    UMulOverflowOpPattern(),
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
    CallOpPattern(),
]
