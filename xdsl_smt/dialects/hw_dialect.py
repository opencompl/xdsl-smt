from typing import Annotated
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.ir import Dialect, OpResult
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    attr_def,
    result_def,
    irdl_op_definition,
)


@irdl_op_definition
class ConstantOp(IRDLOperation):
    """A constant integer value."""

    name = "hw.constant"

    T = Annotated[IntegerType, ConstraintVar("T")]

    value: IntegerAttr[T] = attr_def(IntegerAttr[T])
    result: OpResult = result_def(T)

    def __init__(self, value: IntegerAttr[T]):
        super().__init__(result_types=[value.type], attributes={"value": value})


HW = Dialect(
    "hw",
    [
        ConstantOp,
    ],
)
