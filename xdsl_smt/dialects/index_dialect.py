"""
Defines the index dialect, which represent computations on address-size integers.
This dialect should be moved to xDSL when possible.
"""

from __future__ import annotations
from xdsl.ir import Attribute, Dialect, OpResult
from xdsl.irdl import (
    attr_def,
    operand_def,
    result_def,
    Operand,
    irdl_op_definition,
    IRDLOperation,
)

from ..traits.effects import Pure
from xdsl.dialects.builtin import IndexType


@irdl_op_definition
class Add(IRDLOperation, Pure):
    name = "index.add"
    lhs: Operand = operand_def(IndexType)
    rhs: Operand = operand_def(IndexType)
    result: OpResult = result_def(IndexType)


@irdl_op_definition
class Sub(IRDLOperation, Pure):
    name = "index.sub"
    lhs: Operand = operand_def(IndexType)
    rhs: Operand = operand_def(IndexType)
    result: OpResult = result_def(IndexType)


@irdl_op_definition
class And(IRDLOperation, Pure):
    name = "index.and"
    lhs: Operand = operand_def(IndexType)
    rhs: Operand = operand_def(IndexType)
    result: OpResult = result_def(IndexType)


@irdl_op_definition
class Or(IRDLOperation, Pure):
    name = "index.or"
    lhs: Operand = operand_def(IndexType)
    rhs: Operand = operand_def(IndexType)
    result: OpResult = result_def(IndexType)


@irdl_op_definition
class Xor(IRDLOperation, Pure):
    name = "index.xor"
    lhs: Operand = operand_def(IndexType)
    rhs: Operand = operand_def(IndexType)
    result: OpResult = result_def(IndexType)


@irdl_op_definition
class Cmp(IRDLOperation, Pure):
    name = "index.cmp"
    lhs: Operand = operand_def(IndexType)
    rhs: Operand = operand_def(IndexType)
    predicate: Attribute = attr_def(Attribute)
    result: OpResult = result_def(IndexType)


@irdl_op_definition
class Constant(IRDLOperation, Pure):
    name = "index.constant"
    value: Attribute = attr_def(Attribute)
    result: OpResult = result_def(IndexType)


Index = Dialect(
    "index",
    [
        Add,
        Sub,
        And,
        Or,
        Xor,
        Cmp,
        Constant,
    ],
    [],
)
