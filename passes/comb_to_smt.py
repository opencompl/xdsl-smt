from xdsl.pattern_rewriter import (
    RewritePattern,
    op_type_rewrite_pattern,
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.passes import ModulePass
from xdsl.ir import MLContext, Operation, SSAValue
from xdsl.irdl import IRDLOperation
from xdsl.dialects.builtin import ModuleOp, IntegerType

from .arith_to_smt import FuncToSMTPattern, ReturnPattern, convert_type
from dialects import comb
import dialects.smt_bitvector_dialect as bv_dialect
import dialects.smt_dialect as core_dialect


class ConstantPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: comb.ConstantOp, rewriter: PatternRewriter):
        smt_op = bv_dialect.ConstantOp(op.value)
        rewriter.replace_matched_op(smt_op)


def variadic_op_pattern(
    comb_op_type: type[comb.VariadicCombOp],
    smt_op_type: type[IRDLOperation],
    empty_value: int,
) -> RewritePattern:
    class VariadicOpPattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, comb_op_type):
                return
            res_type = convert_type(op.result.typ)
            assert isinstance(res_type, bv_dialect.BitVectorType)
            if len(op.operands) == 0:
                rewriter.replace_matched_op(
                    bv_dialect.ConstantOp(empty_value, res_type.width)
                )
                return

            current_val = op.operands[0]

            for operand in op.operands[1:]:
                new_op = smt_op_type.build(
                    operands=[current_val, operand],
                    result_types=[convert_type(op.result.typ)],
                )
                current_val = new_op.results[0]
                rewriter.insert_op_before_matched_op(new_op)

            rewriter.replace_matched_op([], [current_val])

    return VariadicOpPattern()


def trivial_binop_pattern(
    comb_op_type: type[comb.BinCombOp], smt_op_type: type[IRDLOperation]
) -> RewritePattern:
    class TrivialBinOpPattern(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, comb_op_type):
                return
            new_op = smt_op_type.build(
                operands=op.operands,
                result_types=[convert_type(op.result.typ)],
            )
            rewriter.replace_matched_op([new_op])
            return super().match_and_rewrite(op, rewriter)

    return TrivialBinOpPattern()


class ICmpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: comb.ICmpOp, rewriter: PatternRewriter) -> None:
        if op.predicate.value.data == 0:
            rewriter.replace_matched_op(core_dialect.EqOp(op.lhs, op.rhs))
            return
        if op.predicate.value.data == 1:
            rewriter.replace_matched_op(core_dialect.DistinctOp(op.lhs, op.rhs))
            return
        if op.predicate.value.data == 2:
            rewriter.replace_matched_op(bv_dialect.SltOp(op.lhs, op.rhs))
            return
        if op.predicate.value.data == 3:
            rewriter.replace_matched_op(bv_dialect.SleOp(op.lhs, op.rhs))
            return
        if op.predicate.value.data == 4:
            rewriter.replace_matched_op(bv_dialect.SgtOp(op.lhs, op.rhs))
            return
        if op.predicate.value.data == 5:
            rewriter.replace_matched_op(bv_dialect.SgeOp(op.lhs, op.rhs))
            return
        if op.predicate.value.data == 6:
            rewriter.replace_matched_op(bv_dialect.UltOp(op.lhs, op.rhs))
            return
        if op.predicate.value.data == 7:
            rewriter.replace_matched_op(bv_dialect.UleOp(op.lhs, op.rhs))
            return
        if op.predicate.value.data == 8:
            rewriter.replace_matched_op(bv_dialect.UgtOp(op.lhs, op.rhs))
            return
        if op.predicate.value.data == 9:
            rewriter.replace_matched_op(bv_dialect.UgeOp(op.lhs, op.rhs))
            return
        raise NotImplementedError()


class ParityPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: comb.ParityOp, rewriter: PatternRewriter) -> None:
        assert isinstance(op.input.typ, bv_dialect.BitVectorType)
        assert op.input.typ.width.data > 0
        bits: list[SSAValue] = []

        for i in range(op.input.typ.width.data):
            extract = bv_dialect.ExtractOp(op.input, i, i)
            bits.append(extract.results[0])
            rewriter.insert_op_before_matched_op(extract)

        res = bits[0]
        for bit in bits[1:]:
            xor_op = bv_dialect.XorOp(res, bit)
            rewriter.insert_op_before_matched_op(xor_op)
            res = xor_op.results[0]

        rewriter.replace_matched_op([], [res])


class ExtractPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: comb.ExtractOp, rewriter: PatternRewriter) -> None:
        start = op.low_bit.value.data
        assert isinstance(op.result, IntegerType)
        end = op.result.width.data + start - 1
        rewriter.replace_matched_op(bv_dialect.ExtractOp(op.input, start, end))


class ConcatPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: comb.ConcatOp, rewriter: PatternRewriter) -> None:
        assert len(op.operands) > 0

        current_val = op.operands[0]

        for operand in op.operands[1:]:
            new_op = bv_dialect.ConcatOp(current_val, operand)
            current_val = new_op.results[0]
            rewriter.insert_op_before_matched_op(new_op)

        rewriter.replace_matched_op([], [current_val])


class ReplicatePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: comb.ReplicateOp, rewriter: PatternRewriter
    ) -> None:
        assert isinstance(op.input.typ, IntegerType)
        assert isinstance(op.result.typ, IntegerType)
        num_repetition = op.result.typ.width.data // op.input.typ.width.data
        current_val = op.operands[0]

        for _ in range(num_repetition - 1):
            new_op = bv_dialect.ConcatOp(current_val, op.operands[0])
            current_val = new_op.results[0]
            rewriter.insert_op_before_matched_op(new_op)

        rewriter.replace_matched_op([], [current_val])


class MuxPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: comb.MuxOp, rewriter: PatternRewriter) -> None:
        rewriter.replace_matched_op(
            [core_dialect.IteOp(op.cond, op.true_value, op.false_value)]
        )


comb_to_smt_patterns: list[RewritePattern] = [
    ConstantPattern(),
    variadic_op_pattern(comb.AddOp, bv_dialect.AddOp, 0),
    variadic_op_pattern(comb.MulOp, bv_dialect.MulOp, 1),
    trivial_binop_pattern(comb.DivUOp, bv_dialect.UDivOp),
    trivial_binop_pattern(comb.DivSOp, bv_dialect.SDivOp),
    trivial_binop_pattern(comb.ModUOp, bv_dialect.URemOp),
    trivial_binop_pattern(comb.ModSOp, bv_dialect.SRemOp),
    trivial_binop_pattern(comb.ShlOp, bv_dialect.ShlOp),
    trivial_binop_pattern(comb.ShrUOp, bv_dialect.LShrOp),
    trivial_binop_pattern(comb.ShrSOp, bv_dialect.AShrOp),
    trivial_binop_pattern(comb.SubOp, bv_dialect.SubOp),
    variadic_op_pattern(comb.OrOp, bv_dialect.OrOp, 0),
    variadic_op_pattern(comb.AndOp, bv_dialect.AndOp, 1),
    variadic_op_pattern(comb.XorOp, bv_dialect.XorOp, 0),
    ICmpPattern(),
    ParityPattern(),
    ExtractPattern(),
    ConcatPattern(),
    ReplicatePattern(),
    MuxPattern(),
]


class CombToSMT(ModulePass):
    name = "comb-to-smt"

    def apply(self, ctx: MLContext, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [*comb_to_smt_patterns, FuncToSMTPattern(), ReturnPattern()]
            )
        )
        walker.rewrite_module(op)
