from typing import Sequence
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType
import xdsl.dialects.arith as arith
from xdsl.utils.hints import isa

from ...dialects import smt_dialect
from ...dialects import smt_bitvector_dialect as bv_dialect
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
    def match_and_rewrite(self, op: arith.AndI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.AndOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class OriRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.OrI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.OrOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class XoriRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.XOrI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.XorOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class ShliRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShLI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.ShlOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class DivsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivSI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.SDivOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class DivuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.DivUI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.UDivOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class RemsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.RemSI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.SRemOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class RemuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.RemUI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.URemOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class ShrsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShRSI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.AShrOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class ShruiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ShRUI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = bv_dialect.LShrOp(operands[0], operands[1])
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class MaxsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MaxSI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        sgt_op = bv_dialect.SgtOp(operands[0], operands[1])
        ite_op = smt_dialect.IteOp(SSAValue.get(sgt_op), operands[0], operands[1])
        res_op = utils_dialect.PairOp(ite_op.res, poison)
        rewriter.replace_matched_op([sgt_op, ite_op, res_op])


class MaxuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MaxUI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        ugt_op = bv_dialect.UgtOp(operands[0], operands[1])
        ite_op = smt_dialect.IteOp(SSAValue.get(ugt_op), operands[0], operands[1])
        res_op = utils_dialect.PairOp(ite_op.res, poison)
        rewriter.replace_matched_op([ugt_op, ite_op, res_op])


class MinsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MinSI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        sle_op = bv_dialect.SleOp(operands[0], operands[1])
        ite_op = smt_dialect.IteOp(SSAValue.get(sle_op), operands[0], operands[1])
        res_op = utils_dialect.PairOp(ite_op.res, poison)
        rewriter.replace_matched_op([sle_op, ite_op, res_op])


class MinuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.MinUI, rewriter: PatternRewriter) -> None:
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
        predicate = op.predicate.value.data
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
        cond_val, cond_poi = get_int_value_and_poison(op.cond, rewriter)
        tr_val, tr_poi = get_int_value_and_poison(op.lhs, rewriter)
        fls_val, fls_poi = get_int_value_and_poison(op.rhs, rewriter)
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
    def match_and_rewrite(self, op: arith.TruncIOp, rewriter: PatternRewriter) -> None:
        val, poison = get_int_value_and_poison(op.input, rewriter)
        assert isinstance(op.result.type, IntegerType)
        new_width = op.result.type.width.data
        res = bv_dialect.ExtractOp(val, new_width - 1, 0)
        res_op = utils_dialect.PairOp(res.res, poison)
        rewriter.replace_matched_op([res, res_op])


class ExtuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ExtUIOp, rewriter: PatternRewriter) -> None:
        val, poison = get_int_value_and_poison(op.input, rewriter)
        assert isinstance(op.result.type, IntegerType)
        assert isinstance(val.type, bv_dialect.BitVectorType)
        new_width = op.result.type.width.data
        old_width = val.type.width.data
        prefix = bv_dialect.ConstantOp(0, new_width - old_width)
        res = bv_dialect.ConcatOp(prefix.res, val)
        res_op = utils_dialect.PairOp(res.res, poison)
        rewriter.replace_matched_op([prefix, res, res_op])


class ExtsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ExtSIOp, rewriter: PatternRewriter) -> None:
        val, poison = get_int_value_and_poison(op.input, rewriter)
        assert isinstance(op.result.type, IntegerType)
        assert isinstance(val.type, bv_dialect.BitVectorType)
        old_width = val.type.width.data
        new_width = op.result.type.width.data
        sign = bv_dialect.ExtractOp(val, old_width - 1, old_width - 1)
        prefix = bv_dialect.RepeatOp(sign.res, new_width - old_width)
        res = bv_dialect.ConcatOp(prefix.res, val)
        res_op = utils_dialect.PairOp(res.res, poison)
        rewriter.replace_matched_op([sign, prefix, res, res_op])


class CeildivuiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CeilDivUI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        assert isinstance(operands[1].type, bv_dialect.BitVectorType)
        width = operands[1].type.width.data
        div_op = bv_dialect.UDivOp(operands[0], operands[1])
        remainder_op = bv_dialect.URemOp(operands[0], operands[1])
        zero = bv_dialect.ConstantOp(0, width)
        one = bv_dialect.ConstantOp(1, width)
        is_zero = smt_dialect.EqOp(zero.res, remainder_op.res)
        one_if_zero = smt_dialect.IteOp(is_zero.res, zero.res, one.res)
        value_op = bv_dialect.AddOp(div_op.res, one_if_zero.res)
        res_op = utils_dialect.PairOp(value_op.res, poison)
        rewriter.replace_matched_op(
            [div_op, remainder_op, zero, one, is_zero, one_if_zero, value_op, res_op]
        )


class CeildivsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.CeilDivSI, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        assert isinstance(operands[1].type, bv_dialect.BitVectorType)
        width = operands[1].type.width.data

        # Check for underflow
        minimum_value = bv_dialect.ConstantOp(2 ** (width - 1), width)
        minus_one = bv_dialect.ConstantOp(2**width - 1, width)
        one = bv_dialect.ConstantOp(1, width)
        lhs_is_min_val = smt_dialect.EqOp(operands[0], minimum_value.res)
        rhs_is_minus_one = smt_dialect.EqOp(operands[1], minus_one.res)
        is_underflow = smt_dialect.AndOp(lhs_is_min_val.res, rhs_is_minus_one.res)

        # Check for division by zero
        zero = bv_dialect.ConstantOp(0, width)
        is_div_by_zero = smt_dialect.EqOp(zero.res, operands[1])

        # Poison if underflow or division by zero or previous poison
        introduce_poison = smt_dialect.OrOp(is_underflow.res, is_div_by_zero.res)
        new_poison = smt_dialect.OrOp(introduce_poison.res, poison)

        # Compute a division rounded by zero
        value_op = bv_dialect.SDivOp(operands[0], operands[1])

        # If the result is positive, add 1 if the remainder is not 0
        is_positive = bv_dialect.SgtOp(value_op.res, zero.res)
        remainder = bv_dialect.SRemOp(operands[0], operands[1])
        is_remainder_not_zero = smt_dialect.DistinctOp(remainder.res, zero.res)
        add_one = bv_dialect.AddOp(value_op.res, one.res)
        should_add_one = smt_dialect.AndOp(is_positive.res, is_remainder_not_zero.res)
        res_value_op = smt_dialect.IteOp(should_add_one.res, add_one.res, value_op.res)
        res_op = utils_dialect.PairOp(res_value_op.res, new_poison.res)

        rewriter.replace_matched_op(
            [
                minimum_value,
                minus_one,
                one,
                lhs_is_min_val,
                rhs_is_minus_one,
                is_underflow,
                zero,
                is_div_by_zero,
                introduce_poison,
                new_poison,
                value_op,
                is_positive,
                remainder,
                is_remainder_not_zero,
                add_one,
                should_add_one,
                res_value_op,
                res_op,
            ]
        )


class FloordivsiRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.FloorDivSI, rewriter: PatternRewriter
    ) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        assert isinstance(operands[1].type, bv_dialect.BitVectorType)
        width = operands[1].type.width.data

        # Check for underflow
        minimum_value = bv_dialect.ConstantOp(2 ** (width - 1), width)
        minus_one = bv_dialect.ConstantOp(2**width - 1, width)
        lhs_is_min_val = smt_dialect.EqOp(operands[0], minimum_value.res)
        rhs_is_minus_one = smt_dialect.EqOp(operands[1], minus_one.res)
        is_underflow = smt_dialect.AndOp(lhs_is_min_val.res, rhs_is_minus_one.res)

        # Check for division by zero
        zero = bv_dialect.ConstantOp(0, width)
        is_div_by_zero = smt_dialect.EqOp(zero.res, operands[1])

        # Poison if underflow or division by zero or previous poison
        introduce_poison = smt_dialect.OrOp(is_underflow.res, is_div_by_zero.res)
        new_poison = smt_dialect.OrOp(introduce_poison.res, poison)

        # Compute a division rounded by zero
        value_op = bv_dialect.SDivOp(operands[0], operands[1])

        # If the result is negative, subtract 1 if the remainder is not 0
        is_negative = bv_dialect.SltOp(value_op.res, zero.res)
        remainder = bv_dialect.SRemOp(operands[0], operands[1])
        is_remainder_not_zero = smt_dialect.DistinctOp(remainder.res, zero.res)
        subtract_one = bv_dialect.AddOp(value_op.res, minus_one.res)
        should_subtract_one = smt_dialect.AndOp(
            is_negative.res, is_remainder_not_zero.res
        )
        res_value_op = smt_dialect.IteOp(
            should_subtract_one.res, subtract_one.res, value_op.res
        )
        res_op = utils_dialect.PairOp(res_value_op.res, new_poison.res)

        rewriter.replace_matched_op(
            [
                minimum_value,
                minus_one,
                lhs_is_min_val,
                rhs_is_minus_one,
                is_underflow,
                zero,
                is_div_by_zero,
                introduce_poison,
                new_poison,
                value_op,
                is_negative,
                remainder,
                is_remainder_not_zero,
                subtract_one,
                should_subtract_one,
                res_value_op,
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
