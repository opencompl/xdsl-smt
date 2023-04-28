from typing import Annotated
from xdsl.ir import Dialect, OpResult
from xdsl.irdl import IRDLOperation, Operand, VarOperand, irdl_op_definition
from xdsl.dialects.pdl import AttributeType, OperationType, TypeType, ValueType


@irdl_op_definition
class OperandOp(IRDLOperation):
    """Match an operand which has a known bit pattern."""

    name = "pdl.kb.operand"

    type: Annotated[Operand, TypeType]
    value: Annotated[OpResult, ValueType]
    attribute: Annotated[OpResult, AttributeType]


@irdl_op_definition
class Attach(IRDLOperation):
    """Attach known bit patterns to an operation."""

    name = "pdl.kb.attach"

    op: Annotated[Operand, OperationType]
    attrs: Annotated[VarOperand, AttributeType]


@irdl_op_definition
class AddOp(IRDLOperation):
    """Add two known bit patterns."""

    name = "pdl.kb.add"

    lhs: Annotated[Operand, AttributeType]
    rhs: Annotated[Operand, AttributeType]
    res: Annotated[OpResult, AttributeType]


PDLKnownBitsDialect = Dialect([OperandOp, Attach, AddOp])
