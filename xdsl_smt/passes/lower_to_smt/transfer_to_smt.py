from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar
from xdsl.irdl import IRDLOperation
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.ir import Operation, SSAValue

from ...dialects import smt_bitvector_dialect as smt_bv
from ...dialects import smt_dialect as smt
from ...dialects import transfer
from xdsl.ir import Attribute
from xdsl_smt.passes.lower_to_smt.smt_rewrite_patterns import SMTLoweringRewritePattern
from xdsl_smt.passes.lower_to_smt.smt_lowerer import SMTLowerer
from xdsl.dialects import func
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
        result: PairType[Attribute, Attribute] = PairType(
            SMTLowerer.lower_type(type.get_fields()[-1]), BoolType()
        )
        for ty in reversed(type.get_fields()[:-1]):
            result = PairType[Attribute, Attribute](SMTLowerer.lower_type(ty), result)
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
) -> SMTLoweringRewritePattern:
    class TrivialBinOpPattern(SMTLoweringRewritePattern):
        def rewrite(
            self,
            op: Operation,
            effect_state: SSAValue | None,
            rewriter: PatternRewriter,
            smt_lowerer: SMTLowerer,
        ) -> SSAValue | None:
            assert isinstance(op, match_type)
            # TODO: How to handle multiple results, or results with different types?
            new_op = rewrite_type.create(
                operands=op.operands,
                result_types=[op.operands[0].type],
            )
            rewriter.replace_matched_op([new_op])
            return effect_state

    return TrivialBinOpPattern()


_OpType = TypeVar("_OpType", bound=Operation)


class SMTPureLoweringPattern(Generic[_OpType], SMTLoweringRewritePattern):
    op_type: type[_OpType]

    @abstractmethod
    def rewrite_pure(
        self,
        op: _OpType,
        rewriter: PatternRewriter,
    ):
        ...

    def rewrite(
        self: SMTPureLoweringPattern[_OpType],
        op: Operation,
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
        smt_lowerer: SMTLowerer,
    ) -> SSAValue | None:
        assert isinstance(op, self.op_type)
        self.rewrite_pure(op, rewriter)
        return effect_state


def smt_pure_lowering_pattern(
    op_type: type[_OpType],
) -> type[SMTPureLoweringPattern[_OpType]]:
    class SMTPureLoweringPatternImpl(SMTPureLoweringPattern[op_type]):
        def __init__(self):
            self.op_type = op_type

    return SMTPureLoweringPatternImpl


class UMulOverflowOpPattern(smt_pure_lowering_pattern(transfer.UMulOverflowOp)):
    def rewrite_pure(
        self,
        op: transfer.UMulOverflowOp,
        rewriter: PatternRewriter,
    ):
        assert isinstance(op, transfer.UMulOverflowOp)
        suml_nooverflow = smt_bv.UmulNoOverflowOp.get(op.operands[0], op.operands[1])

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(suml_nooverflow.res, b1.res, b0.res)
        poison_op = smt.ConstantBoolOp.from_bool(False)
        res = PairOp(bool_to_bv.res, poison_op.res)
        rewriter.replace_matched_op(
            [suml_nooverflow, b0, b1, bool_to_bv, poison_op, res]
        )


class CmpOpPattern(smt_pure_lowering_pattern(transfer.CmpOp)):
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

    def rewrite_pure(self, op: transfer.CmpOp, rewriter: PatternRewriter) -> None:
        predicate = op.predicate.value.data

        rewrite_type = self.new_ops[predicate]
        new_op = rewrite_type.create(operands=op.operands, result_types=[BoolType()])

        resList: list[Operation] = [new_op]

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
        # return resList


class GetOpPattern(smt_pure_lowering_pattern(transfer.GetOp)):
    def rewrite_pure(self, op: transfer.GetOp, rewriter: PatternRewriter) -> None:
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


class MakeOpPattern(smt_pure_lowering_pattern(transfer.MakeOp)):
    def rewrite_pure(self, op: transfer.MakeOp, rewriter: PatternRewriter) -> None:
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


class GetBitWidthOpPattern(smt_pure_lowering_pattern(transfer.GetBitWidthOp)):
    """1. There is no direct API to obtain the bit width of a vector.
    2. All bit width are fixed values.
    As a result, we replace this method with a constant."""

    def rewrite_pure(
        self, op: transfer.GetBitWidthOp, rewriter: PatternRewriter
    ) -> None:
        assert isinstance(op.op.type, smt_bv.BitVectorType)
        width = op.op.type.width
        value = width
        rewriter.replace_matched_op(smt_bv.ConstantOp(value, width))


class ConstantOpPattern(smt_pure_lowering_pattern(transfer.Constant)):
    def rewrite_pure(self, op: transfer.Constant, rewriter: PatternRewriter) -> None:
        assert isinstance(op.op.type, smt_bv.BitVectorType)
        width = op.op.type.width
        value = op.value.value.data
        rewriter.replace_matched_op(smt_bv.ConstantOp(value, width))


class CountLOneOpPattern(smt_pure_lowering_pattern(transfer.CountLOneOp)):
    def rewrite_pure(self, op: transfer.CountLOneOp, rewriter: PatternRewriter) -> None:
        operand = op.operands[0]
        resList = count_lones(operand)
        for newOp in resList[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.replace_matched_op(resList[-1])


class CountLZeroOpPattern(smt_pure_lowering_pattern(transfer.CountLZeroOp)):
    def rewrite_pure(
        self, op: transfer.CountLZeroOp, rewriter: PatternRewriter
    ) -> None:
        operand = op.operands[0]
        resList = count_lzeros(operand)
        for newOp in resList[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.replace_matched_op(resList[-1])


class CountROneOpPattern(smt_pure_lowering_pattern(transfer.CountROneOp)):
    def rewrite_pure(self, op: transfer.CountROneOp, rewriter: PatternRewriter) -> None:
        operand = op.operands[0]
        resList, afterList = count_rones(operand)
        for newOp in resList[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        for newOp in afterList:
            rewriter.insert_op_after_matched_op(newOp)
        rewriter.replace_matched_op(resList[-1])


class CountRZeroOpPattern(smt_pure_lowering_pattern(transfer.CountRZeroOp)):
    def rewrite_pure(
        self, op: transfer.CountRZeroOp, rewriter: PatternRewriter
    ) -> None:
        operand = op.operands[0]
        resList, afterList = count_rzeros(operand)
        for newOp in resList[:-1]:
            rewriter.insert_op_after_matched_op(newOp)
        for newOp in afterList:
            rewriter.insert_op_after_matched_op(newOp)
        # countRzero is different, it insert operation after the matched op.
        rewriter.replace_matched_op(resList[-1])


class SetHighBitsOpPattern(smt_pure_lowering_pattern(transfer.SetHighBitsOp)):
    def rewrite_pure(
        self, op: transfer.SetHighBitsOp, rewriter: PatternRewriter
    ) -> None:
        result = set_high_bits(op.operands[0], op.operands[1])
        for newOp in result[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.replace_matched_op(result[-1])


class GetLowBitsOpPattern(smt_pure_lowering_pattern(transfer.GetLowBitsOp)):
    def rewrite_pure(
        self, op: transfer.GetLowBitsOp, rewriter: PatternRewriter
    ) -> None:
        result = get_low_bits(op.operands[0], op.operands[1])
        for newOp in result[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.replace_matched_op(result[-1])


class SMinOpPattern(smt_pure_lowering_pattern(transfer.SMinOp)):
    def rewrite_pure(self, op: transfer.SMinOp, rewriter: PatternRewriter) -> None:
        smin_if = smt_bv.UleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        rewriter.replace_matched_op(
            smt.IteOp(smin_if.results[0], op.operands[0], op.operands[1])
        )


class SMaxOpPattern(smt_pure_lowering_pattern(transfer.SMaxOp)):
    def rewrite_pure(self, op: transfer.SMaxOp, rewriter: PatternRewriter) -> None:
        smax_if = smt_bv.SleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(smax_if)
        rewriter.replace_matched_op(
            smt.IteOp(smax_if.results[0], op.operands[1], op.operands[0])
        )


class UMaxOpPattern(smt_pure_lowering_pattern(transfer.UMaxOp)):
    def rewrite_pure(self, op: transfer.UMaxOp, rewriter: PatternRewriter) -> None:
        umax_if = smt_bv.UleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(umax_if)
        rewriter.replace_matched_op(
            smt.IteOp(umax_if.results[0], op.operands[1], op.operands[0])
        )


class UMinOpPattern(smt_pure_lowering_pattern(transfer.UMinOp)):
    def rewrite_pure(self, op: transfer.UMinOp, rewriter: PatternRewriter) -> None:
        umin_if = smt_bv.UleOp(op.operands[0], op.operands[1])
        rewriter.insert_op_before_matched_op(umin_if)
        rewriter.replace_matched_op(
            smt.IteOp(umin_if.results[0], op.operands[0], op.operands[1])
        )


class CallOpPattern(smt_pure_lowering_pattern(func.CallOp)):
    def rewrite_pure(self, op: func.CallOp, rewriter: PatternRewriter) -> None:
        module = op.parent_op()
        callee = op.callee.string_value()
        while not isinstance(module, ModuleOp):
            assert module is not None
            module = module.parent_op()
        isinstance(module, ModuleOp)
        for funcOp in module.ops:
            if isinstance(funcOp, DefineFunOp):
                name = funcOp.fun_name
                if name is not None and name.data == callee:
                    newCallOp = CallOp.get(funcOp.results[0], op.arguments)
                    rewriter.replace_matched_op(newCallOp)
                    return
        assert False and "Cannot find the desired call"


transfer_to_smt_patterns: dict[type[Operation], SMTLoweringRewritePattern] = {
    transfer.AndOp: trivial_pattern(transfer.AndOp, smt_bv.AndOp),
    transfer.OrOp: trivial_pattern(transfer.OrOp, smt_bv.OrOp),
    transfer.XorOp: trivial_pattern(transfer.XorOp, smt_bv.XorOp),
    transfer.SubOp: trivial_pattern(transfer.SubOp, smt_bv.SubOp),
    transfer.AddOp: trivial_pattern(transfer.AddOp, smt_bv.AddOp),
    transfer.MulOp: trivial_pattern(transfer.MulOp, smt_bv.MulOp),
    transfer.NegOp: trivial_pattern(transfer.NegOp, smt_bv.NotOp),
    transfer.UMulOverflowOp: UMulOverflowOpPattern(),
    transfer.CmpOp: CmpOpPattern(),
    transfer.GetOp: GetOpPattern(),
    transfer.MakeOp: MakeOpPattern(),
    transfer.Constant: ConstantOpPattern(),
    transfer.GetBitWidthOp: GetBitWidthOpPattern(),
    transfer.CountLOneOp: CountLOneOpPattern(),
    transfer.CountLZeroOp: CountLZeroOpPattern(),
    transfer.CountROneOp: CountROneOpPattern(),
    transfer.CountRZeroOp: CountRZeroOpPattern(),
    transfer.SetHighBitsOp: SetHighBitsOpPattern(),
    transfer.GetLowBitsOp: GetLowBitsOpPattern(),
    transfer.SMinOp: SMinOpPattern(),
    transfer.SMaxOp: SMaxOpPattern(),
    transfer.UMaxOp: UMaxOpPattern(),
    transfer.UMinOp: UMinOpPattern(),
    CallOp: CallOpPattern(),
}
