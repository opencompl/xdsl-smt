from __future__ import annotations

from typing import Sequence
from functools import cache

from xdsl.ir import (
    SSAValue,
    Region,
    Attribute,
    Block,
    Operation,
    BlockArgument,
    OpResult,
)
from xdsl.builder import Builder
from xdsl.rewriter import InsertPoint
from xdsl.dialects.builtin import StringAttr
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.pdl import TypeOp, AttributeOp, OperandOp, ResultOp, OperationOp


def func_to_pdl(
    func: FuncOp, *, arguments: Sequence[SSAValue | None] | None = None
) -> tuple[Region, tuple[SSAValue, ...], SSAValue | None, tuple[SSAValue, ...]]:
    """
    Creates a region containing PDL instructions corresponding to this
    program. Returns a containing `Region`, together with the values of the
    inputs, the root operation (i.e., the operation the first returned value
    comes from, if it exists), and the returned values.
    """

    if arguments is not None:
        assert len(arguments) == len(func.function_type.inputs)

    body = Region(Block())
    builder = Builder(InsertPoint.at_end(body.block))

    @cache
    def get_type(ty: Attribute) -> SSAValue:
        return builder.insert(TypeOp(ty)).results[0]

    @cache
    def get_attribute(attr: Attribute) -> SSAValue:
        return builder.insert(AttributeOp(attr)).results[0]

    @cache
    def get_argument(arg: int) -> SSAValue:
        if arguments is not None:
            if (value := arguments[arg]) is not None:
                return value
        return builder.insert(OperandOp(get_type(func.args[arg].type))).results[0]

    ops: dict[Operation, SSAValue] = {}

    def get_value(value: SSAValue) -> SSAValue:
        match value:
            case BlockArgument(index=k):
                return get_argument(k)
            case OpResult(op=op, index=i):
                return builder.insert(ResultOp(i, ops[op])).results[0]
            case x:
                raise ValueError(f"Unknown value: {x}")

    [*operations, ret] = list(func.body.ops)
    for op in operations:
        pattern = builder.insert(
            OperationOp(
                op.name,
                [StringAttr(s) for s in op.properties.keys()],
                [get_value(operand) for operand in op.operands],
                # Iteration order on dict is consistent betwen `.keys()` and
                # `.values()`.
                [get_attribute(prop) for prop in op.properties.values()],
                [get_type(ty) for ty in op.result_types],
            )
        )
        ops[op] = pattern.results[0]

    assert isinstance(ret, ReturnOp)
    assert len(ret.operands) == 1
    root = ret.operands[0]

    return (
        body,
        tuple(get_argument(i) for i in range(len(func.function_type.inputs))),
        ops[root.op] if isinstance(root, OpResult) else None,
        tuple(get_value(res) for res in ret.operands),
    )
