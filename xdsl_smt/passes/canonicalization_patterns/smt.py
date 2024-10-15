from xdsl.ir import OpResult, SSAValue, Operation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl_smt.dialects import smt_dialect as smt, smt_bitvector_dialect as smt_bv


def get_bool_constant(value: SSAValue) -> bool | None:
    if not isinstance(value, OpResult):
        return None
    if not isinstance((constant := value.op), smt.ConstantBoolOp):
        return None
    return constant.value.data


def get_bv_constant(value: SSAValue) -> int | None:
    if not isinstance(value, OpResult):
        return None
    if not isinstance((constant := value.op), smt_bv.ConstantOp):
        return None
    return constant.value.value.data


class QuantifierCanonicalizationPattern(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # forall x. True -> True
        # forall x. False -> False
        # exists x. True -> True
        # exists x. False -> False
        if isinstance(op, smt.ForallOp | smt.ExistsOp):
            if (value := get_bool_constant(op.return_val)) is None:
                return None
            rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(value))
            return


class NotCanonicalizationPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: smt.NotOp, rewriter: PatternRewriter):
        # not True -> False
        # not False -> True
        if (value := get_bool_constant(op.arg)) is None:
            return None
        rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(not value))
        return


class ImpliesCanonicalizationPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: smt.ImpliesOp, rewriter: PatternRewriter):
        # True => x -> x
        # False => x -> True
        if (value := get_bool_constant(op.lhs)) is not None:
            if value:
                rewriter.replace_matched_op([], [op.rhs])
            else:
                rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(True))
            return
        # x => True -> True
        # x => False -> not x
        if (value := get_bool_constant(op.rhs)) is not None:
            if value:
                rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(True))
            else:
                rewriter.replace_matched_op(smt.NotOp.get(op.lhs))
            return
        # x => x -> True
        if op.lhs == op.rhs:
            rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(True))
            return


class AndCanonicalizationPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: smt.AndOp, rewriter: PatternRewriter):
        # True && x -> x
        # False && x -> False
        if (value := get_bool_constant(op.lhs)) is not None:
            if value:
                rewriter.replace_matched_op([], [op.rhs])
            else:
                rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(False))
            return
        # x && True -> x
        # x && False -> False
        if (value := get_bool_constant(op.rhs)) is not None:
            if value:
                rewriter.replace_matched_op([], [op.lhs])
            else:
                rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(False))
            return
        # x && x -> x
        if op.lhs == op.rhs:
            rewriter.replace_matched_op([], [op.lhs])
            return


class OrCanonicalizationPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: smt.OrOp, rewriter: PatternRewriter):
        # True || x -> True
        # False || x -> x
        if (value := get_bool_constant(op.lhs)) is not None:
            if value:
                rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(True))
            else:
                rewriter.replace_matched_op([], [op.rhs])
            return
        # x || True -> True
        # x || False -> x
        if (value := get_bool_constant(op.rhs)) is not None:
            if value:
                rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(True))
            else:
                rewriter.replace_matched_op([], [op.lhs])
            return
        # x || x -> x
        if op.lhs == op.rhs:
            rewriter.replace_matched_op([], [op.lhs])
            return


class XorCanonicalizationPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: smt.XorOp, rewriter: PatternRewriter):
        # True ^ x -> not x
        # False ^ x -> x
        if (value := get_bool_constant(op.lhs)) is not None:
            if value:
                rewriter.replace_matched_op(smt.NotOp.get(op.rhs))
            else:
                rewriter.replace_matched_op([], [op.rhs])
            return
        # x ^ True -> not True
        # x ^ False -> x
        if (value := get_bool_constant(op.rhs)) is not None:
            if value:
                rewriter.replace_matched_op(smt.NotOp.get(op.lhs))
            else:
                rewriter.replace_matched_op([], [op.lhs])
            return
        # x ^ x -> False
        if op.lhs == op.rhs:
            rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(False))
            return


class EqCanonicalizationPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: smt.EqOp, rewriter: PatternRewriter):
        # True == x -> x
        # False == x -> not x
        if (value := get_bool_constant(op.lhs)) is not None:
            if value:
                rewriter.replace_matched_op([], [op.rhs])
            else:
                rewriter.replace_matched_op(smt.NotOp.get(op.rhs))
            return
        # x == True -> x
        # x == False -> not x
        if (value := get_bool_constant(op.rhs)) is not None:
            if value:
                rewriter.replace_matched_op([], [op.lhs])
            else:
                rewriter.replace_matched_op(smt.NotOp.get(op.lhs))
            return
        # x == x -> True
        if op.lhs == op.rhs:
            rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(True))
            return
        # Constant folding for bitvectors
        if (value := get_bv_constant(op.lhs)) is not None:
            if (value2 := get_bv_constant(op.rhs)) is not None:
                rewriter.replace_matched_op(
                    smt.ConstantBoolOp.from_bool(value == value2)
                )
                return


class DistinctCanonicalizationPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: smt.DistinctOp, rewriter: PatternRewriter):
        # True != x -> not x
        # False != x -> x
        if (value := get_bool_constant(op.lhs)) is not None:
            if value:
                rewriter.replace_matched_op(smt.NotOp.get(op.rhs))
            else:
                rewriter.replace_matched_op([], [op.rhs])
            return
        # x != True -> not x
        # x != False -> x
        if (value := get_bool_constant(op.rhs)) is not None:
            if value:
                rewriter.replace_matched_op(smt.NotOp.get(op.lhs))
            else:
                rewriter.replace_matched_op([], [op.lhs])
            return
        # x != x -> False
        if op.lhs == op.rhs:
            rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(False))
            return
        # Constant folding for bitvectors
        if (value := get_bv_constant(op.lhs)) is not None:
            if (value2 := get_bv_constant(op.rhs)) is not None:
                rewriter.replace_matched_op(
                    smt.ConstantBoolOp.from_bool(value != value2)
                )
                return


class IteCanonicalizationPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: smt.IteOp, rewriter: PatternRewriter):
        # x if True else y -> x
        # x if False else y -> y
        if (value := get_bool_constant(op.cond)) is not None:
            rewriter.replace_matched_op([], [op.true_val if value else op.false_val])
            return
        # x if y else x -> x
        if op.true_val == op.false_val:
            rewriter.replace_matched_op([], [op.true_val])
            return