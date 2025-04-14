"""
Defines the `hw.constant` operation. This should be moved to xDSL when possible.
"""

from typing import ClassVar
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import Dialect, OpResult
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    result_def,
    irdl_op_definition,
    VarConstraint,
    base,
)


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """A constant integer value."""

    name = "hw.constant"

    T: ClassVar = VarConstraint("_Range", base(IntegerType))

    value = attr_def(IntegerAttr.constr(type=T))
    result: OpResult = result_def(T)

    def __init__(self, value: IntegerAttr[IntegerType]):
        super().__init__(result_types=[value.type], attributes={"value": value})


HW = Dialect(
    "hw",
    [
        ConstantOp,
    ],
)
