from __future__ import annotations
from xdsl.ir import Attribute, Dialect, OpResult
from xdsl.irdl import OpAttr, Operand, irdl_op_definition, IRDLOperation

from traits.effects import Pure
from xdsl.dialects.builtin import IndexType
from typing import Annotated


@irdl_op_definition
class Add(IRDLOperation, Pure):
    name = "index.add"
    lhs: Annotated[Operand, IndexType]
    rhs: Annotated[Operand, IndexType]
    result: Annotated[OpResult, IndexType]


@irdl_op_definition
class And(IRDLOperation, Pure):
    name = "index.and"
    lhs: Annotated[Operand, IndexType]
    rhs: Annotated[Operand, IndexType]
    result: Annotated[OpResult, IndexType]


@irdl_op_definition
class Cmp(IRDLOperation, Pure):
    name = "index.cmp"
    lhs: Annotated[Operand, IndexType]
    rhs: Annotated[Operand, IndexType]
    predicate: OpAttr[Attribute]
    result: Annotated[OpResult, IndexType]


@irdl_op_definition
class Constant(IRDLOperation, Pure):
    name = "index.constant"
    value: OpAttr[Attribute]
    result: Annotated[OpResult, IndexType]


Index = Dialect(
    [
        Add,
        And,
        Cmp,
        Constant,
    ],
    [],
)
