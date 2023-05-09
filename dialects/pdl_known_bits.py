from typing import Annotated
from xdsl.dialects.builtin import i32
from xdsl.ir import Dialect, OpResult
from xdsl.irdl import IRDLOperation, Operand, VarOperand, irdl_op_definition
from xdsl.dialects.pdl import AttributeType, OperationType, TypeType, ValueType


@irdl_op_definition
class OperandOp(IRDLOperation):
    """Match an operand which has a known bit pattern."""

    name = "pdl.kb.operand"

    type: Annotated[Operand, TypeType]
    value: Annotated[OpResult, ValueType]
    zeros: Annotated[OpResult, i32]
    ones: Annotated[OpResult, i32]


@irdl_op_definition
class Attach(IRDLOperation):
    """Attach known bit patterns to an operation."""

    name = "pdl.kb.attach"

    op: Annotated[Operand, OperationType]
    zeros: Annotated[Operand, i32]
    ones: Annotated[Operand, i32]


PDLKnownBitsDialect = Dialect([OperandOp, Attach])
