from xdsl.dialects.builtin import FunctionType, IntegerType
from xdsl.dialects.llvm import LLVMVoidType
from xdsl.ir import SSAValue
from xdsl.parser import IntegerAttr
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa
import xdsl_smt.dialects.llvm_dialect as llvm
import xdsl_smt.dialects.smt_dialect as smt
import xdsl_smt.dialects.smt_bitvector_dialect as smt_bv
import xdsl_smt.dialects.smt_utils_dialect as smt_utils
from xdsl_smt.passes.lower_to_smt.arith_to_smt import (
    reduce_poison_values,
    get_int_value_and_poison,
)
from xdsl_smt.passes.lower_to_smt.lower_to_smt import LowerToSMT


class ReturnPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.ReturnOp, rewriter: PatternRewriter):
        if not op.arg:
            rewriter.replace_matched_op([])
            return

        smt_op = smt.ReturnOp(op.arg)
        rewriter.replace_matched_op([smt_op])


class FuncToSMTPattern(RewritePattern):
    """Convert llvm.func to an SMT formula"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.FuncOp, rewriter: PatternRewriter):
        # We only handle single-block regions for now
        if len(op.body.blocks) != 1:
            raise Exception("Cannot convert multi-block functions")

        operand_types = [
            LowerToSMT.lower_type(input) for input in op.function_type.inputs.data
        ]
        if op.function_type.output == LLVMVoidType():
            raise Exception("Cannot convert void functions")
        result_type = LowerToSMT.lower_types(op.function_type.output)

        # The SMT function replacing the func.func function
        smt_func = smt.DefineFunOp.from_function_type(
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


class AddRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.AddOp, rewriter: PatternRewriter) -> None:
        if op.get_attr_or_prop("nsw") or op.get_attr_or_prop("nuw"):
            raise Exception("Cannot handle nsw/nuw add")
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = smt_bv.AddOp(operands[0], operands[1])
        res_op = smt_utils.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class SubRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.SubOp, rewriter: PatternRewriter) -> None:
        if op.get_attr_or_prop("nsw") or op.get_attr_or_prop("nuw"):
            raise Exception("Cannot handle nsw/nuw sub")
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = smt_bv.SubOp(operands[0], operands[1])
        res_op = smt_utils.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class MulRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.MulOp, rewriter: PatternRewriter) -> None:
        if op.get_attr_or_prop("nsw") or op.get_attr_or_prop("nuw"):
            raise Exception("Cannot handle nsw/nuw mul")
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = smt_bv.MulOp(operands[0], operands[1])
        res_op = smt_utils.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class UDivRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.UDivOp, rewriter: PatternRewriter) -> None:
        if op.get_attr_or_prop("exact"):
            raise Exception("Cannot handle exact udiv")
        operands, poison = reduce_poison_values(op.operands, rewriter)
        assert isinstance(op.res.type, IntegerType)
        width = op.res.type.width.data

        value_op = smt_bv.UDivOp(operands[0], operands[1])

        # Check for division by zero
        zero = smt_bv.ConstantOp(0, width)
        is_rhs_zero = smt.EqOp(operands[1], zero.res)
        new_poison = smt.OrOp(is_rhs_zero.res, poison)

        res_op = smt_utils.PairOp(value_op.res, new_poison.res)
        rewriter.replace_matched_op([value_op, zero, is_rhs_zero, new_poison, res_op])


class SDivRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.SDivOp, rewriter: PatternRewriter) -> None:
        if op.get_attr_or_prop("exact"):
            raise Exception("Cannot handle exact sdiv")
        operands, poison = reduce_poison_values(op.operands, rewriter)

        assert isinstance(op.res.type, IntegerType)
        width = op.res.type.width.data

        # Check for division by zero
        zero = smt_bv.ConstantOp(0, width)
        is_div_by_zero = smt.EqOp(zero.res, operands[1])

        # Check for underflow
        minimum_value = smt_bv.ConstantOp(2 ** (width - 1), width)
        minus_one = smt_bv.ConstantOp(2**width - 1, width)
        lhs_is_min_val = smt.EqOp(operands[0], minimum_value.res)
        rhs_is_minus_one = smt.EqOp(operands[1], minus_one.res)
        is_underflow = smt.AndOp(lhs_is_min_val.res, rhs_is_minus_one.res)

        # New poison cases
        introduce_poison = smt.OrOp(is_div_by_zero.res, is_underflow.res)
        new_poison = smt.OrOp(introduce_poison.res, poison)

        value_op = smt_bv.SDivOp(operands[0], operands[1])
        res_op = smt_utils.PairOp(value_op.res, new_poison.res)
        rewriter.replace_matched_op(
            [
                zero,
                is_div_by_zero,
                minimum_value,
                minus_one,
                lhs_is_min_val,
                rhs_is_minus_one,
                is_underflow,
                introduce_poison,
                new_poison,
                value_op,
                res_op,
            ]
        )


class URemOpRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.URemOp, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        assert isinstance(op.res.type, IntegerType)
        width = op.res.type.width.data

        value_op = smt_bv.URemOp(operands[0], operands[1])

        # Poison if the rhs is zero
        zero = smt_bv.ConstantOp(0, width)
        is_rhs_zero = smt.EqOp(operands[1], zero.res)
        new_poison = smt.OrOp(is_rhs_zero.res, poison)

        res_op = smt_utils.PairOp(value_op.res, new_poison.res)
        rewriter.replace_matched_op([value_op, zero, is_rhs_zero, new_poison, res_op])


class SRemRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.SRemOp, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        assert isinstance(op.res.type, IntegerType)
        width = op.res.type.width.data

        value_op = smt_bv.SRemOp(operands[0], operands[1])

        # Poison if the rhs is zero
        zero = smt_bv.ConstantOp(0, width)
        is_rhs_zero = smt.EqOp(operands[1], zero.res)
        new_poison = smt.OrOp(is_rhs_zero.res, poison)

        res_op = smt_utils.PairOp(value_op.res, new_poison.res)
        rewriter.replace_matched_op([value_op, zero, is_rhs_zero, new_poison, res_op])


class AndRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.AndOp, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = smt_bv.AndOp(operands[0], operands[1])
        res_op = smt_utils.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class OrRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.OrOp, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = smt_bv.OrOp(operands[0], operands[1])
        res_op = smt_utils.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class XorRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.XOrOp, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = smt_bv.XorOp(operands[0], operands[1])
        res_op = smt_utils.PairOp(value_op.res, poison)
        rewriter.replace_matched_op([value_op, res_op])


class ShlRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.ShlOp, rewriter: PatternRewriter) -> None:
        if op.get_attr_or_prop("nuw") or op.get_attr_or_prop("nsw"):
            raise Exception("Cannot handle nuw/nsw shl")
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = smt_bv.ShlOp(operands[0], operands[1])

        # If the shift amount is greater than the width of the value, poison
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width.data
        width_op = smt_bv.ConstantOp(width, width)
        shift_amount_too_big = smt_bv.UgtOp(operands[1], width_op.res)
        new_poison = smt.OrOp(shift_amount_too_big.res, poison)

        res_op = smt_utils.PairOp(value_op.res, new_poison.res)
        rewriter.replace_matched_op(
            [value_op, width_op, shift_amount_too_big, new_poison, res_op]
        )


class AShrRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.AShrOp, rewriter: PatternRewriter) -> None:
        if op.get_attr_or_prop("exact"):
            raise Exception("Cannot handle exact ashr")
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = smt_bv.AShrOp(operands[0], operands[1])

        # If the shift amount is greater than the width of the value, poison
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width.data
        width_op = smt_bv.ConstantOp(width, width)
        shift_amount_too_big = smt_bv.UgtOp(operands[1], width_op.res)
        new_poison = smt.OrOp(shift_amount_too_big.res, poison)

        res_op = smt_utils.PairOp(value_op.res, new_poison.res)
        rewriter.replace_matched_op(
            [value_op, width_op, shift_amount_too_big, new_poison, res_op]
        )


class LShrRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.LShrOp, rewriter: PatternRewriter) -> None:
        if op.get_attr_or_prop("exact"):
            raise Exception("Cannot handle exact lshr")
        operands, poison = reduce_poison_values(op.operands, rewriter)
        value_op = smt_bv.LShrOp(operands[0], operands[1])

        # If the shift amount is greater than the width of the value, poison
        assert isinstance(operands[0].type, smt_bv.BitVectorType)
        width = operands[0].type.width.data
        width_op = smt_bv.ConstantOp(width, width)
        shift_amount_too_big = smt_bv.UgtOp(operands[1], width_op.res)
        new_poison = smt.OrOp(shift_amount_too_big.res, poison)

        res_op = smt_utils.PairOp(value_op.res, new_poison.res)
        rewriter.replace_matched_op(
            [value_op, width_op, shift_amount_too_big, new_poison, res_op]
        )


class ICmpOpRewritePattern(RewritePattern):
    predicates = [
        smt.EqOp,
        smt.DistinctOp,
        smt_bv.UgtOp,
        smt_bv.UgeOp,
        smt_bv.UltOp,
        smt_bv.UleOp,
        smt_bv.SgtOp,
        smt_bv.SgeOp,
        smt_bv.SltOp,
        smt_bv.SleOp,
    ]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.ICmpOp, rewriter: PatternRewriter) -> None:
        operands, poison = reduce_poison_values(op.operands, rewriter)
        predicate = op.predicate.value.data
        sgt_op = self.predicates[predicate](operands[0], operands[1])
        bv_0 = smt_bv.ConstantOp(0, 1)
        bv_1 = smt_bv.ConstantOp(1, 1)
        ite_op = smt.IteOp(SSAValue.get(sgt_op), bv_1.res, bv_0.res)
        res_op = smt_utils.PairOp(ite_op.res, poison)
        rewriter.replace_matched_op([sgt_op, bv_0, bv_1, ite_op, res_op])


class SelectRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.SelectOp, rewriter: PatternRewriter) -> None:
        cond_val, cond_poi = get_int_value_and_poison(op.cond, rewriter)
        tr_val, tr_poi = get_int_value_and_poison(op.lhs, rewriter)
        fls_val, fls_poi = get_int_value_and_poison(op.rhs, rewriter)
        bv_0 = smt_bv.ConstantOp(
            1,
            1,
        )
        to_smt_bool = smt.EqOp(cond_val, bv_0.res)
        res_val = smt.IteOp(to_smt_bool.res, tr_val, fls_val)
        br_poi = smt.IteOp(to_smt_bool.res, tr_poi, fls_poi)
        res_poi = smt.IteOp(cond_poi, cond_poi, br_poi.res)
        res_op = smt_utils.PairOp(res_val.res, res_poi.res)
        rewriter.replace_matched_op(
            [bv_0, to_smt_bool, res_val, br_poi, res_poi, res_op]
        )


class IntegerConstantRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: llvm.ConstantOp, rewriter: PatternRewriter):
        if not isa(op.value, IntegerAttr[IntegerType]):
            raise Exception("Cannot convert constant of type that are not integer type")
        value_op = smt_bv.ConstantOp(op.value)
        poison_op = smt.ConstantBoolOp.from_bool(False)
        res_op = smt_utils.PairOp(value_op.res, poison_op.res)
        rewriter.replace_matched_op([value_op, poison_op, res_op])


llvm_to_smt_patterns: list[RewritePattern] = [
    FuncToSMTPattern(),
    ReturnPattern(),
    AddRewritePattern(),
    SubRewritePattern(),
    MulRewritePattern(),
    UDivRewritePattern(),
    SDivRewritePattern(),
    URemOpRewritePattern(),
    SRemRewritePattern(),
    AndRewritePattern(),
    OrRewritePattern(),
    XorRewritePattern(),
    ShlRewritePattern(),
    AShrRewritePattern(),
    LShrRewritePattern(),
    ICmpOpRewritePattern(),
    SelectRewritePattern(),
    IntegerConstantRewritePattern(),
]
