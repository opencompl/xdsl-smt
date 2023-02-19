"""
This pass expect takes care of:
- Removing dead code
- Constant folding
- Folding expressions (i.e. (x, y).first -> x)
for the SMT dialects
"""

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import MLContext, OpResult, Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern
import dialects.smt_dialect as smt

from passes.dead_code_elimination import dead_code_elimination


class FoldCorePattern(RewritePattern):

    @staticmethod
    def get_constant(value: SSAValue) -> bool | None:
        if not isinstance(value, OpResult):
            return None
        if not isinstance((constant := value.op), smt.ConstantBoolOp):
            return None
        return constant.value.data

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # forall x. True -> True
        # forall x. False -> False
        # exists x. True -> True
        # exists x. False -> False
        # TODO: Check that `x` is inhabited.
        # If not, then the forall/exists is always true/false.
        if isinstance(op, smt.ForallOp | smt.ExistsOp):
            if (value := self.get_constant(op.return_val)) is None:
                return None
            rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(value))
            return

        # not True -> False
        # not False -> True
        if isinstance(op, smt.NotOp):
            if (value := self.get_constant(op.arg)) is None:
                return None
            rewriter.replace_matched_op(
                smt.ConstantBoolOp.from_bool(not value))
            return

        # True => x -> x
        # False => x -> True
        # x => True -> True
        # x => False -> not x
        # x => x -> True
        if isinstance(op, smt.ImpliesOp):
            if (value := self.get_constant(op.lhs)) is not None:
                if value:
                    rewriter.replace_matched_op([], [op.rhs])
                else:
                    rewriter.replace_matched_op(
                        smt.ConstantBoolOp.from_bool(True))
                return
            if (value := self.get_constant(op.rhs)) is not None:
                if value:
                    rewriter.replace_matched_op(
                        smt.ConstantBoolOp.from_bool(True))
                else:
                    rewriter.replace_matched_op(smt.NotOp.get(op.lhs))
                return
            if op.lhs == op.rhs:
                rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(True))
                return
            return

        # True && x -> x
        # False && x -> False
        # x && True -> x
        # x && False -> False
        # x && x -> x
        if isinstance(op, smt.AndOp):
            if (value := self.get_constant(op.lhs)) is not None:
                if value:
                    rewriter.replace_matched_op([], [op.rhs])
                else:
                    rewriter.replace_matched_op(
                        smt.ConstantBoolOp.from_bool(False))
                return
            if (value := self.get_constant(op.rhs)) is not None:
                if value:
                    rewriter.replace_matched_op([], [op.lhs])
                else:
                    rewriter.replace_matched_op(
                        smt.ConstantBoolOp.from_bool(False))
                return
            if op.lhs == op.rhs:
                rewriter.replace_matched_op([], [op.lhs])
                return
            return

        # True || x -> True
        # False || x -> x
        # x || True -> True
        # x || False -> x
        # x || x -> x
        if isinstance(op, smt.OrOp):
            if (value := self.get_constant(op.lhs)) is not None:
                if value:
                    rewriter.replace_matched_op(
                        smt.ConstantBoolOp.from_bool(True))
                else:
                    rewriter.replace_matched_op([], [op.rhs])
                return
            if (value := self.get_constant(op.rhs)) is not None:
                if value:
                    rewriter.replace_matched_op(
                        smt.ConstantBoolOp.from_bool(True))
                else:
                    rewriter.replace_matched_op([], [op.lhs])
                return
            if op.lhs == op.rhs:
                rewriter.replace_matched_op([], [op.lhs])
                return
            return

        # True ^ x -> not x
        # False ^ x -> x
        # x ^ True -> not True
        # x ^ False -> x
        # x ^ x -> False
        if isinstance(op, smt.XorOp):
            if (value := self.get_constant(op.lhs)) is not None:
                if value:
                    rewriter.replace_matched_op(smt.NotOp.get(op.rhs))
                else:
                    rewriter.replace_matched_op([], [op.rhs])
                return
            if (value := self.get_constant(op.rhs)) is not None:
                if value:
                    rewriter.replace_matched_op(smt.NotOp.get(op.lhs))
                else:
                    rewriter.replace_matched_op([], [op.lhs])
                return
            if op.lhs == op.rhs:
                rewriter.replace_matched_op(
                    smt.ConstantBoolOp.from_bool(False))
                return
            return

        # True == x -> x
        # False == x -> not x
        # x == True -> x
        # x == False -> not x
        # x == x -> True
        if isinstance(op, smt.EqOp):
            if (value := self.get_constant(op.lhs)) is not None:
                if value:
                    rewriter.replace_matched_op([], [op.rhs])
                else:
                    rewriter.replace_matched_op(smt.NotOp.get(op.rhs))
                return
            if (value := self.get_constant(op.rhs)) is not None:
                if value:
                    rewriter.replace_matched_op([], [op.lhs])
                else:
                    rewriter.replace_matched_op(smt.NotOp.get(op.lhs))
                return
            if op.lhs == op.rhs:
                rewriter.replace_matched_op(smt.ConstantBoolOp.from_bool(True))
                return
            return

        # True != x -> not x
        # False != x -> x
        # x != True -> not x
        # x != False -> x
        # x != x -> False
        if isinstance(op, smt.DiscinctOp):
            if (value := self.get_constant(op.lhs)) is not None:
                if value:
                    rewriter.replace_matched_op(smt.NotOp.get(op.rhs))
                else:
                    rewriter.replace_matched_op([], [op.rhs])
                return
            if (value := self.get_constant(op.rhs)) is not None:
                if value:
                    rewriter.replace_matched_op(smt.NotOp.get(op.lhs))
                else:
                    rewriter.replace_matched_op([], [op.lhs])
                return
            if op.lhs == op.rhs:
                rewriter.replace_matched_op(
                    smt.ConstantBoolOp.from_bool(False))
                return
            return

        # x if True else y -> x
        # x if False else y -> y
        # x if y else x -> x
        if isinstance(op, smt.IteOp):
            if (value := self.get_constant(op.cond)) is not None:
                rewriter.replace_matched_op(
                    [], [op.true_val if value else op.false_val])
                return
            if op.true_val == op.false_val:
                rewriter.replace_matched_op([], [op.true_val])
                return
            return


constant_fold_core_patterns = []


def canonicalize_smt(ctx: MLContext, module: ModuleOp):
    walker = PatternRewriteWalker(FoldCorePattern())
    walker.rewrite_module(module)

    # Finish with dead code elimination
    dead_code_elimination(ctx, module)
