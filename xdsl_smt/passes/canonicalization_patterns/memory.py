"""This file defines simple canonicalization patterns for the memory dialect."""

from typing import TypeVar
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl_smt.dialects import memory_dialect as mem

OpT = TypeVar(
    "OpT", bound=mem.SetBlockBytesOp | mem.SetBlockLiveMarkerOp | mem.SetBlockSizeOp
)


class GetSetMemoryPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.GetMemoryOp, rewriter: PatternRewriter):
        if not isinstance(set_memory := op.state.owner, mem.SetMemoryOp):
            return None
        rewriter.replace_matched_op([], [set_memory.memory])


class GetSetBlockPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.GetBlockOp, rewriter: PatternRewriter):
        if not isinstance(set_block := op.memory.owner, mem.SetBlockOp):
            return None
        if set_block.block_id != op.block_id:
            return None
        rewriter.replace_matched_op([], [set_block.block])


class SetSetBlockPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.SetBlockOp, rewriter: PatternRewriter):
        if not isinstance(set_block := op.memory.owner, mem.SetBlockOp):
            return None
        if set_block.block_id != op.block_id:
            return None
        if len(set_block.res.uses) != 1:
            return None
        new_op = mem.SetBlockOp(op.block, set_block.memory, set_block.block_id)
        rewriter.replace_matched_op([new_op])
        rewriter.erase_op(set_block)


def traverse_block_modifications(
    block: SSAValue, expected_op_type: type[OpT], depth: int = 3
) -> OpT | None:
    """
    Traverse the chain of block modifications starting from the given block.
    Returns the first operation of the expected type found in the chain.
    """
    while depth > 0 and isinstance(
        owner := block.owner,
        mem.SetBlockBytesOp | mem.SetBlockLiveMarkerOp | mem.SetBlockSizeOp,
    ):
        if isinstance(block.owner, expected_op_type):
            return block.owner
        block = owner.memory_block
        depth -= 1
    return None


class GetSetBlockSizePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.GetBlockSizeOp, rewriter: PatternRewriter):
        last_set_op = traverse_block_modifications(op.memory_block, mem.SetBlockSizeOp)
        if last_set_op is None:
            return None
        rewriter.replace_matched_op([], [last_set_op.size])


class GetSetBlockBytesPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.GetBlockBytesOp, rewriter: PatternRewriter):
        last_set_op = traverse_block_modifications(op.memory_block, mem.SetBlockBytesOp)
        if last_set_op is None:
            return None
        rewriter.replace_matched_op([], [last_set_op.bytes])


class GetSetBlockLiveMarkerPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: mem.GetBlockLiveMarkerOp, rewriter: PatternRewriter
    ):
        last_set_op = traverse_block_modifications(
            op.memory_block, mem.SetBlockLiveMarkerOp
        )
        if last_set_op is None:
            return None
        rewriter.replace_matched_op([], [last_set_op.live])
