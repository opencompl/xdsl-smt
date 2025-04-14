"""Define utilities to rewrite operations."""

from typing import Iterable

from xdsl.ir import Operation, OpResult


def new_ops(op: Operation) -> Iterable[Operation]:
    """Iterate over an un-parented operation and its operands in post-order (operands first)"""

    if op.parent is None:
        for child_op in op.operands:
            if isinstance(child_op, OpResult):
                yield from new_ops(child_op.op)
        yield op
