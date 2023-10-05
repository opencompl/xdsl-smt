from typing import Sequence
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.utils.hints import isa

from ...dialects import smt_dialect
from ...dialects import smt_bitvector_dialect as bv_dialect
from ...dialects import arith_dialect as arith
from ...dialects import smt_utils_dialect as utils_dialect


def get_int_value_and_poison(
    val: SSAValue, rewriter: PatternRewriter
) -> tuple[SSAValue, SSAValue]:
    value = utils_dialect.FirstOp(val)
    poison = utils_dialect.SecondOp(val)
    rewriter.insert_op_before_matched_op([value, poison])
    return value.res, poison.res


class IntegerConstantRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter):
        if not isa(op.value, IntegerAttr[IntegerType]):
            raise Exception("Cannot convert constant of type that are not integer type")
        value_op = bv_dialect.ConstantOp(op.value)
        poison_op = smt_dialect.ConstantBoolOp.from_bool(False)
        res_op = utils_dialect.PairOp(value_op.res, poison_op.res)
        rewriter.replace_matched_op([value_op, poison_op, res_op])


def reduce_poison_values(
    operands: Sequence[SSAValue], rewriter: PatternRewriter
) -> tuple[Sequence[SSAValue], SSAValue]:
    assert len(operands) == 2

    left_value, left_poison = get_int_value_and_poison(operands[0], rewriter)
    right_value, right_poison = get_int_value_and_poison(operands[1], rewriter)
    res_poison_op = smt_dialect.OrOp(left_poison, right_poison)
    rewriter.insert_op_before_matched_op(res_poison_op)
    return [left_value, right_value], res_poison_op.res


class AddiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.AddOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class SubiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Subi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.SubOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class MuliRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Muli, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.MulOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class AndiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Andi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.AndOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class OriRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Ori, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.OrOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class XoriRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Xori, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.XorOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class ShliRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Shli, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.ShlOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class DivsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Divsi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.SDivOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class DivuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Divui, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.UDivOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class RemsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Remsi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.SRemOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class RemuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Remui, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.URemOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class ShrsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Shrsi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.AShrOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class ShruiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Shrui, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.LShrOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class MaxsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Maxsi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        sgt_op = bv_dialect.SgtOp(operands[0], operands[1])
        ite_op = smt_dialect.IteOp(SSAValue.get(sgt_op), operands[0], operands[1])
        res_op = utils_dialect.PairOp(ite_op.res, poison)
        rewriter.replace_matched_op([sgt_op, ite_op, res_op])


class MaxuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Maxui, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        ugt_op = bv_dialect.UgtOp(operands[0], operands[1])
        ite_op = smt_dialect.IteOp(SSAValue.get(ugt_op), operands[0], operands[1])
        res_op = utils_dialect.PairOp(ite_op.res, poison)
        rewriter.replace_matched_op([ugt_op, ite_op, res_op])


class MinsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Minsi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        sle_op = bv_dialect.SleOp(operands[0], operands[1])
        ite_op = smt_dialect.IteOp(SSAValue.get(sle_op), operands[0], operands[1])
        res_op = utils_dialect.PairOp(ite_op.res, poison)
        rewriter.replace_matched_op([sle_op, ite_op, res_op])


class MinuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Minui, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        ule_op = bv_dialect.UleOp(operands[0], operands[1])
        ite_op = smt_dialect.IteOp(SSAValue.get(ule_op), operands[0], operands[1])
        res_op = utils_dialect.PairOp(ite_op.res, poison)
        rewriter.replace_matched_op([ule_op, ite_op, res_op])


class CmpiRewritePattern(RewritePattern):
    predicates = [
        smt_dialect.EqOp,
        smt_dialect.DistinctOp,
        bv_dialect.SltOp,
        bv_dialect.SleOp,
        bv_dialect.SgtOp,
        bv_dialect.SgeOp,
        bv_dialect.UltOp,
        bv_dialect.UleOp,
        bv_dialect.UgtOp,
        bv_dialect.UgeOp,
    ]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Cmpi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        predicate = op.attributes["predicate"].value.data  # type: int
        sgt_op = self.predicates[predicate](operands[0], operands[1])
        bv_0 = bv_dialect.ConstantOp(0, 1)
        bv_1 = bv_dialect.ConstantOp(1, 1)
        ite_op = smt_dialect.IteOp(SSAValue.get(sgt_op), bv_1.res, bv_0.res)
        res_op = utils_dialect.PairOp(ite_op.res, poison)
        rewriter.replace_matched_op([sgt_op, bv_0, bv_1, ite_op, res_op])


class SelectRewritePattern(RewritePattern):
    """
    select poison a, b -> poison
    select true, a, poison -> a
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Select, rewriter: PatternRewriter) -> None:
        cond_val, cond_poi = get_int_value_and_poison(op.condition, rewriter)
        tr_val, tr_poi = get_int_value_and_poison(op.true_value, rewriter)
        fls_val, fls_poi = get_int_value_and_poison(op.false_value, rewriter)
        bv_0 = bv_dialect.ConstantOp(
            1,
            1,
        )
        to_smt_bool = smt_dialect.EqOp(cond_val, bv_0.res)
        res_val = smt_dialect.IteOp(to_smt_bool.res, tr_val, fls_val)
        br_poi = smt_dialect.IteOp(to_smt_bool.res, tr_poi, fls_poi)
        res_poi = smt_dialect.IteOp(cond_poi, cond_poi, br_poi.res)
        res_op = utils_dialect.PairOp(res_val.res, res_poi.res)
        rewriter.replace_matched_op(
            [bv_0, to_smt_bool, res_val, br_poi, res_poi, res_op]
        )


class TrunciRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Trunci, rewriter: PatternRewriter) -> None:
        val, poison = get_int_value_and_poison(op._in, rewriter)
        assert isinstance(op.out.type, bv_dialect.BitVectorType)
        new_width = op.out.type.width.data  # type: int
        res = bv_dialect.ExtractOp(val, new_width - 1, 0)
        res_op = utils_dialect.PairOp(res.res, poison)
        rewriter.replace_matched_op([res, res_op])


class ExtuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Extui, rewriter: PatternRewriter) -> None:
        val, poison = get_int_value_and_poison(op._in, rewriter)
        assert isinstance(op.out.type, bv_dialect.BitVectorType)
        new_width = op.out.type.width.data  # type: int
        old_width = val.type.width.data  # type: int
        prefix = bv_dialect.ConstantOp(0, new_width - old_width)
        res = bv_dialect.ConcatOp(prefix.res, val)
        res_op = utils_dialect.PairOp(res.res, poison)
        rewriter.replace_matched_op([prefix, res, res_op])


class ExtsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Extsi, rewriter: PatternRewriter) -> None:
        val, poison = get_int_value_and_poison(op._in, rewriter)
        assert isinstance(op.out.type, bv_dialect.BitVectorType)
        old_width = val.type.width.data  # type: int
        new_width = op.out.type.width.data  # type: int
        sign = bv_dialect.ExtractOp(val, old_width - 1, old_width - 1)
        prefix = bv_dialect.RepeatOp(sign.res, new_width - old_width)
        res = bv_dialect.ConcatOp(prefix.res, val)
        res_op = utils_dialect.PairOp(res.res, poison)
        rewriter.replace_matched_op([sign, prefix, res, res_op])


# Ceil unsigned div: add (divisor - 1)
class CeildivuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Ceildivui, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        assert isinstance(operands[1].type, bv_dialect.BitVectorType)
        bv_1 = bv_dialect.ConstantOp(1, operands[1].type.width)
        divisor_sub_1 = bv_dialect.SubOp(operands[1], bv_1.res)
        dividend_add = bv_dialect.AddOp(operands[0], divisor_sub_1.res)
        value_op = bv_dialect.UDivOp(dividend_add.res, operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op(
            [bv_1, divisor_sub_1, dividend_add, value_op, res_op]
        )


# Ceil signed div: likewise, but only when sign bit unset
class CeildivsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Ceildivsi, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        assert isinstance(operands[1].type, bv_dialect.BitVectorType)
        width = operands[1].type.width.data  # type: int
        bv_1 = bv_dialect.ConstantOp(1, width)
        divisor_sub_1 = bv_dialect.SubOp(operands[1], bv_1.res)
        dividend_add = bv_dialect.AddOp(operands[0], divisor_sub_1.res)
        sign = bv_dialect.ExtractOp(operands[0], width - 1, width - 1)
        bv_0_1 = bv_dialect.ConstantOp(0, 1)
        is_0 = smt_dialect.EqOp(sign.res, bv_0_1.res)
        ite_op = smt_dialect.IteOp(is_0.res, dividend_add.res, operands[0])
        value_op = bv_dialect.SDivOp(ite_op.res, operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op(
            [
                bv_1,
                divisor_sub_1,
                dividend_add,
                sign,
                bv_0_1,
                is_0,
                ite_op,
                value_op,
                res_op,
            ]
        )


# Floor signed div: subtract (divisor - 1) when sign bit set
class FloordivsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.Floordivsi, rewriter: PatternRewriter
    ) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        assert isinstance(operands[1].type, bv_dialect.BitVectorType)
        width = operands[1].type.width.data  # type: int
        bv_1 = bv_dialect.ConstantOp(1, width)
        divisor_sub_1 = bv_dialect.SubOp(operands[1], bv_1.res)
        dividend_sub = bv_dialect.SubOp(operands[0], divisor_sub_1.res)
        sign = bv_dialect.ExtractOp(operands[0], width - 1, width - 1)
        bv_1_1 = bv_dialect.ConstantOp(1, 1)
        is_1 = smt_dialect.EqOp(sign.res, bv_1_1.res)
        ite_op = smt_dialect.IteOp(is_1.res, dividend_sub.res, operands[0])
        value_op = bv_dialect.SDivOp(ite_op.res, operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op(
            [
                bv_1,
                divisor_sub_1,
                dividend_sub,
                sign,
                bv_1_1,
                is_1,
                ite_op,
                value_op,
                res_op,
            ]
        )


arith_to_smt_patterns: list[RewritePattern] = [
    IntegerConstantRewritePattern(),
    AddiRewritePattern(),
    SubiRewritePattern(),
    MuliRewritePattern(),
    AndiRewritePattern(),
    OriRewritePattern(),
    XoriRewritePattern(),
    ShliRewritePattern(),
    DivuiRewritePattern(),
    DivsiRewritePattern(),
    RemuiRewritePattern(),
    RemsiRewritePattern(),
    ShruiRewritePattern(),
    ShrsiRewritePattern(),
    MinuiRewritePattern(),
    MinsiRewritePattern(),
    MaxuiRewritePattern(),
    MaxsiRewritePattern(),
    CmpiRewritePattern(),
    SelectRewritePattern(),
    TrunciRewritePattern(),
    ExtuiRewritePattern(),
    ExtsiRewritePattern(),
    CeildivuiRewritePattern(),
    CeildivsiRewritePattern(),
    FloordivsiRewritePattern(),
]
