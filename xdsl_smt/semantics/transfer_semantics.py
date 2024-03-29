from xdsl.irdl import IRDLOperation
from dataclasses import dataclass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.ir import Operation

from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import transfer
from xdsl_smt.passes.lower_to_smt.lower_to_smt import LowerToSMT
from xdsl.dialects.func import Call
from xdsl_smt.dialects.smt_utils_dialect import PairType, SecondOp, FirstOp, PairOp
from xdsl_smt.dialects.smt_dialect import BoolType, DefineFunOp, CallOp
from xdsl_smt.semantics.semantics import OperationSemantics
from xdsl.ir import Operation, Region, SSAValue, Attribute
from xdsl_smt.semantics.builtin_semantics import IntegerAttrSemantics
from typing import Mapping, Sequence
from xdsl.utils.hints import isa
from xdsl.parser import AnyIntegerAttr
from xdsl.dialects.builtin import ModuleOp, IntegerAttr, IndexType
from xdsl_smt.utils.transfer_to_smt_util import (
    get_low_bits,
    set_high_bits,
    set_low_bits,
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


class ConstantOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width
        const_value = attributes["value"]
        if isinstance(const_value, Attribute):
            assert isa(const_value, AnyIntegerAttr)
            const_value = const_value.value.data

        bv_const = smt_bv.ConstantOp(const_value, width)
        rewriter.insert_op_before_matched_op(bv_const)
        return (bv_const.res,)


class GetAllOnesOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width
        const_value = (1 << width.data) - 1

        bv_const = smt_bv.ConstantOp(const_value, width)
        rewriter.insert_op_before_matched_op(bv_const)
        return (bv_const.res,)


class GetBitWidthOpSemantics(OperationSemantics):
    """1. There is no direct API to obtain the bit width of a vector.
    2. All bit width are fixed values.
    As a result, we replace this method with a constant."""

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width
        const_value = width
        bv_const = smt_bv.ConstantOp(const_value, width)
        rewriter.insert_op_before_matched_op(bv_const)
        return (bv_const.res,)


class MakeOpSemantics(OperationSemantics):
    # The last element is useless, getOp won't access it
    # So it can be any bool value
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        false_constant = smt.ConstantBoolOp.from_bool(False)
        argList = operands
        opList: list[Operation] = [PairOp(argList[-1], false_constant.res)]
        result = opList[-1].results[0]
        for ty in reversed(argList[:-1]):
            opList.append(PairOp(ty, result))
            result = opList[-1].results[0]

        rewriter.insert_op_before_matched_op(false_constant)
        for newOp in opList[:-1]:
            rewriter.insert_op_before_matched_op(newOp)
        rewriter.insert_op_before_matched_op(opList[-1])
        return (opList[-1].results[0],)


class GetOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        index = attributes["index"]
        arg = operands[0]
        insertOps: list[Operation] = []
        assert isinstance(index, IntegerAttr)
        index = index.value.data
        while index != 0:
            insertOps.append(SecondOp(arg))
            arg = insertOps[-1].results[0]
            index -= 1
        insertOps.append(FirstOp(arg))
        rewriter.insert_op_before_matched_op(insertOps)

        return (insertOps[-1].results[0],)


@dataclass
class TrivialOpSemantics(OperationSemantics):
    comb_op_type: type[Operation]
    smt_op_type: type[Operation]

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        new_op = self.smt_op_type.create(
            operands=operands,
            result_types=[LowerToSMT.lower_type(results[0])],
        )
        rewriter.insert_op_before_matched_op([new_op])
        return (new_op.results[0],)


class UMulOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        suml_nooverflow = smt_bv.UmulNoOverflowOp.get(operands[0], operands[1])

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(suml_nooverflow.res, b0.res, b1.res)
        poison_op = smt.ConstantBoolOp.from_bool(False)
        res = PairOp(bool_to_bv.res, poison_op.res)
        rewriter.insert_op_before_matched_op(
            [suml_nooverflow, b0, b1, bool_to_bv, poison_op, res]
        )
        return (res.res,)


class ShlOverflowOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        width = operands[0].type.width
        const_value = width
        bv_width = smt_bv.ConstantOp(const_value, width)

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        cmp_op = smt_bv.UgeOp(operands[1], bv_width.res)
        bool_to_bv = smt.IteOp(cmp_op.res, b1.res, b0.res)
        poison_op = smt.ConstantBoolOp.from_bool(False)
        res = PairOp(bool_to_bv.res, poison_op.res)
        rewriter.insert_op_before_matched_op(
            [bv_width, b0, b1, cmp_op, bool_to_bv, poison_op, res]
        )
        return (res.res,)


class IsPowerOf2OpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        width = operands[0].type.width
        const_value = width

        b1 = smt_bv.ConstantOp.from_int_value(1, const_value)
        b0 = smt_bv.ConstantOp.from_int_value(0, const_value)
        b1_1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0_1 = smt_bv.ConstantOp.from_int_value(0, 1)
        op_minus_one = smt_bv.SubOp(operands[0], b1.res)
        and_op = smt_bv.AndOp(operands[0], op_minus_one.res)
        eq_op = smt.EqOp(b0.res, and_op.res)
        bool_to_bv = smt.IteOp(eq_op.res, b1_1.res, b0_1.res)
        poison_op = smt.ConstantBoolOp.from_bool(False)
        res = PairOp(bool_to_bv.res, poison_op.res)
        rewriter.insert_op_before_matched_op(
            [
                b0,
                b1,
                b1_1,
                b0_1,
                op_minus_one,
                and_op,
                eq_op,
                bool_to_bv,
                poison_op,
                res,
            ]
        )
        return (res.res,)


class CmpOpSemantics(OperationSemantics):
    new_ops = [
        smt.EqOp,
        smt.DistinctOp,
        smt_bv.SltOp,
        smt_bv.SleOp,
        smt_bv.SgtOp,
        smt_bv.SgeOp,
        smt_bv.UltOp,
        smt_bv.UleOp,
        smt_bv.UgtOp,
        smt_bv.UgeOp,
    ]

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        predicate = attributes["predicate"]
        assert isinstance(predicate, IntegerAttr)
        predicate = predicate.value.data

        rewrite_type = self.new_ops[predicate]
        new_op = rewrite_type.create(operands=operands, result_types=[BoolType()])

        resList: list[Operation] = [new_op]

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(resList[-1].results[0], b1.results[0], b0.results[0])

        poison_op = smt.ConstantBoolOp.from_bool(False)
        res_op = PairOp(bool_to_bv.results[0], poison_op.res)

        resList += [b1, b0, bool_to_bv, poison_op, res_op]
        rewriter.insert_op_before_matched_op(resList)
        return (res_op.res,)


class IntersectsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        and_res = smt_bv.AndOp(operands[0], operands[1])
        assert isinstance(and_res.res.type, smt_bv.BitVectorType)
        const_0 = smt_bv.ConstantOp(0, and_res.res.type.width)
        eq_0 = smt.EqOp(and_res.res, const_0.res)

        resList: list[Operation] = [and_res, const_0, eq_0]

        b1 = smt_bv.ConstantOp.from_int_value(1, 1)
        b0 = smt_bv.ConstantOp.from_int_value(0, 1)
        bool_to_bv = smt.IteOp(resList[-1].results[0], b0.results[0], b1.results[0])

        poison_op = smt.ConstantBoolOp.from_bool(False)
        res_op = PairOp(bool_to_bv.results[0], poison_op.res)

        resList += [b1, b0, bool_to_bv, poison_op, res_op]
        rewriter.insert_op_before_matched_op(resList)
        return (res_op.res,)


class CountLOneOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        operand = operands[0]
        resList = count_lones(operand)
        rewriter.insert_op_before_matched_op(resList)
        return (resList[-1].results[0],)


class CountLZeroOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        operand = operands[0]
        resList = count_lzeros(operand)
        rewriter.insert_op_before_matched_op(resList)
        return (resList[-1].results[0],)


class CountROneOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        operand = operands[0]
        resList, afterList = count_rones(operand)
        rewriter.insert_op_before_matched_op(resList)
        rewriter.insert_op_before_matched_op(afterList)
        return (resList[-1].results[0],)


class CountRZeroOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        operand = operands[0]
        resList, afterList = count_rzeros(operand)
        rewriter.insert_op_before_matched_op(resList)
        rewriter.insert_op_before_matched_op(afterList)
        return (resList[-1].results[0],)


class SetHighBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        result = set_high_bits(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(result)
        return (result[-1].results[0],)


class SetLowBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        result = set_low_bits(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(result)
        return (result[-1].results[0],)


class GetLowBitsOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        result = get_low_bits(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(result)
        return (result[-1].results[0],)


class SMinOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        smin_if = smt_bv.SleOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return (ite_op.res,)


class UMinOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        smin_if = smt_bv.UleOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return (ite_op.res,)


class SMaxOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        smin_if = smt_bv.SgtOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return (ite_op.res,)


class UMaxOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        smin_if = smt_bv.UgtOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(smin_if)
        ite_op = smt.IteOp(smin_if.results[0], operands[0], operands[1])
        rewriter.insert_op_before_matched_op(ite_op)
        return (ite_op.res,)


class SelectOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        bv1 = smt_bv.ConstantOp.from_int_value(1, 1)
        bv_val = FirstOp(operands[0])
        eq1 = smt.EqOp(bv_val.res, bv1.res)
        ite_op = smt.IteOp(eq1.res, operands[1], operands[2])
        rewriter.insert_op_before_matched_op([bv1, bv_val, eq1, ite_op])
        return (ite_op.res,)


class RepeatOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        cntOp = operands[1].owner
        assert (
            isinstance(cntOp, smt_bv.ConstantOp) and "repeat count has to be a constant"
        )
        cnt = cntOp.value.value.data
        repeatOp = smt_bv.RepeatOp(operands[0], cnt)
        rewriter.insert_op_before_matched_op(repeatOp)
        return (repeatOp.res,)


class ExtractOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        numBitsOp = operands[1].owner
        bitPositionOp = operands[2].owner
        assert (
            isinstance(numBitsOp, smt_bv.ConstantOp) and "num bits has to be a constant"
        )
        assert (
            isinstance(bitPositionOp, smt_bv.ConstantOp)
            and "bit position has to be a constant"
        )
        numBits = numBitsOp.value.value.data
        bitPosition = bitPositionOp.value.value.data
        extractOp = smt_bv.ExtractOp(operands[0], numBits + bitPosition, bitPosition)
        rewriter.insert_op_before_matched_op(extractOp)
        return (extractOp.res,)


class ConstRangeForOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
        lb = operands[0].owner
        ub = operands[1].owner
        step = operands[2].owner
        assert (
            isinstance(lb, smt_bv.ConstantOp)
            and "loop lower bound has to be a constant"
        )
        assert (
            isinstance(ub, smt_bv.ConstantOp)
            and "loop upper bound has to be a constant"
        )
        assert isinstance(step, smt_bv.ConstantOp) and "loop step has to be a constant"
        lb_int = lb.value.value.data
        ub_int = ub.value.value.data
        step_int = step.value.value.data

        assert step_int != 0 and "step size should not be zero"
        if step_int > 0:
            assert (
                ub_int > lb_int
                and "the upper bound should be larger than the lower bound"
            )
        else:
            assert (
                ub_int < lb_int
                and "the upper bound should be smaller than the lower bound"
            )

        iter_args = operands[3:]
        iter_args_num = len(iter_args)

        indvar, *block_iter_args = regions[0].block.args

        value_map: dict[SSAValue, SSAValue] = {}

        value_map[indvar] = operands[0]
        for i in range(iter_args_num):
            value_map[block_iter_args[i]] = iter_args[i]
        last_result = None
        for i in range(lb_int, ub_int, step_int):
            for cur_op in regions[0].block.ops:
                if not isinstance(cur_op, transfer.NextLoopOp):
                    clone_op = cur_op.clone()
                    for idx in range(len(clone_op.operands)):
                        if cur_op.operands[idx] in value_map:
                            clone_op.operands[idx] = value_map[cur_op.operands[idx]]
                    if len(cur_op.results) != 0:
                        value_map[cur_op.results[0]] = clone_op.results[0]
                    rewriter.insert_op_before_matched_op(clone_op)
                    continue
                if isinstance(cur_op, transfer.NextLoopOp):
                    if i + step_int < ub_int:
                        new_value_map: dict[SSAValue, SSAValue] = {}
                        cur_ind = transfer.Constant(operands[1], i + step_int).result
                        new_value_map[indvar] = cur_ind
                        rewriter.insert_op_before_matched_op(cur_ind.owner)
                        for idx, arg in enumerate(block_iter_args):
                            new_value_map[block_iter_args[idx]] = value_map[
                                cur_op.operands[idx]
                            ]
                        value_map = new_value_map
                    else:
                        make_res = [value_map[arg] for arg in cur_op.arguments]
                        assert (
                            len(make_res) == 1
                            and "current we only support for one returned value from for"
                        )
                        last_result = make_res[0]
        return (last_result,)


transfer_semantics: dict[type[Operation], OperationSemantics] = {
    transfer.Constant: ConstantOpSemantics(),
    transfer.AddOp: TrivialOpSemantics(transfer.AddOp, smt_bv.AddOp),
    transfer.MulOp: TrivialOpSemantics(transfer.MulOp, smt_bv.MulOp),
    transfer.OrOp: TrivialOpSemantics(transfer.OrOp, smt_bv.OrOp),
    transfer.AndOp: TrivialOpSemantics(transfer.AndOp, smt_bv.AndOp),
    transfer.XorOp: TrivialOpSemantics(transfer.XorOp, smt_bv.XorOp),
    transfer.SubOp: TrivialOpSemantics(transfer.SubOp, smt_bv.SubOp),
    transfer.NegOp: TrivialOpSemantics(transfer.NegOp, smt_bv.NotOp),
    transfer.ShlOp: TrivialOpSemantics(transfer.ShlOp, smt_bv.ShlOp),
    transfer.ConcatOp: TrivialOpSemantics(transfer.ConcatOp, smt_bv.ConcatOp),
    transfer.RepeatOp: RepeatOpSemantics(),
    transfer.ExtractOp: ExtractOpSemantics(),
    transfer.UMulOverflowOp: UMulOverflowOpSemantics(),
    transfer.ShlOverflowOp: ShlOverflowOpSemantics(),
    transfer.CmpOp: CmpOpSemantics(),
    transfer.GetOp: GetOpSemantics(),
    transfer.MakeOp: MakeOpSemantics(),
    transfer.GetBitWidthOp: GetBitWidthOpSemantics(),
    transfer.CountLOneOp: CountLOneOpSemantics(),
    transfer.CountLZeroOp: CountLZeroOpSemantics(),
    transfer.CountROneOp: CountROneOpSemantics(),
    transfer.CountRZeroOp: CountRZeroOpSemantics(),
    transfer.SMaxOp: SMaxOpSemantics(),
    transfer.SMinOp: SMinOpSemantics(),
    transfer.UMaxOp: UMaxOpSemantics(),
    transfer.UMinOp: UMinOpSemantics(),
    transfer.SetHighBitsOp: SetHighBitsOpSemantics(),
    transfer.SetLowBitsOp: SetLowBitsOpSemantics(),
    transfer.GetLowBitsOp: GetLowBitsOpSemantics(),
    transfer.SelectOp: SelectOpSemantics(),
    transfer.IsPowerOf2Op: IsPowerOf2OpSemantics(),
    transfer.GetAllOnesOp: GetAllOnesOpSemantics(),
    transfer.ConstRangeForOp: ConstRangeForOpSemantics(),
    transfer.IntersectsOp: IntersectsOpSemantics(),
}
