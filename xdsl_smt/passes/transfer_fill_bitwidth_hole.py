from dataclasses import dataclass
from xdsl.pattern_rewriter import (
    op_type_rewrite_pattern,
)

from ..dialects import transfer
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import SSAValue, BlockArgument
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern


def check_dominance(value: SSAValue, constant_ops: list[transfer.Constant]) -> bool:
    """
    Given a SSA value, it returns if the value dominates all constant operations.
    """
    if isinstance(value, BlockArgument):
        return True
    parent_op = value.owner
    for op in constant_ops:
        if op.is_before_in_block(parent_op):
            return False
    return True


def find_dominator(constant_ops: list[transfer.Constant]) -> SSAValue:
    """
    Returns
    -------
    A dominator that dominates all constant operations

    Find a SSA value that dominates all constant operations
    """
    for op in constant_ops:
        for use in op.result.uses:
            for operand in use.operation.operands:
                if operand != op.result and not isinstance(
                    operand.owner, transfer.Constant
                ):
                    if check_dominance(operand, constant_ops):
                        return operand
    assert False and "Cannot find any value to fill bitwidth hole"


@dataclass(frozen=True)
class FillBitwidthHoleOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: transfer.BitwidthHoleOp, rewriter: PatternRewriter
    ) -> None:
        user_constant_ops: list[transfer.Constant] = []
        for use in op.result.uses:
            assert isinstance(use.operation, transfer.Constant)
            user_constant_ops.append(use.operation)
        assert len(user_constant_ops) != 0
        ssa_value = find_dominator(user_constant_ops)

        for op in user_constant_ops:
            op.operands[0] = ssa_value
        rewriter.erase_matched_op()


@dataclass(frozen=True)
class FillBitwidthHole(ModulePass):
    name = "fillBitwidthHole"

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(FillBitwidthHoleOpPattern())
        walker.rewrite_module(op)
