from xdsl.dialects.builtin import i32
from xdsl.ir import Dialect, OpResult
from xdsl.irdl import (
    operand_def,
    result_def,
    IRDLOperation,
    Operand,
    irdl_op_definition,
)
from xdsl.dialects.pdl import OperationType, TypeType, ValueType


@irdl_op_definition
class KBOperandOp(IRDLOperation):
    """Match an operand which has a known bit pattern."""

    name = "pdl.kb.operand"

    type: Operand = operand_def(TypeType)
    value: OpResult = result_def(ValueType)
    zeros: OpResult = result_def(i32)
    ones: OpResult = result_def(i32)


@irdl_op_definition
class KBAttachOp(IRDLOperation):
    """Attach known bit patterns to an operation."""

    name = "pdl.kb.attach"

    op: Operand = operand_def(OperationType)
    zeros: Operand = operand_def(i32)
    ones: Operand = operand_def(i32)


PDLKnownBitsDialect = Dialect([KBOperandOp, KBAttachOp])
