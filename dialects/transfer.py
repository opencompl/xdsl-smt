from __future__ import annotations

from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntegerAttr,
)
from typing import Annotated

from xdsl.ir import ParametrizedAttribute, Dialect, TypeAttribute, OpResult

from xdsl.irdl import (
    OpAttr,
    Operand,
    VarOperand,
    irdl_attr_definition,
    irdl_op_definition,
    ParameterDef,
    IRDLOperation,
)
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class AbstractValueType(ParametrizedAttribute, TypeAttribute):
    name: str = "abs_value"

    fields: ParameterDef[ArrayAttr[IndexType]]

    def get_num_fields(self) -> int:
        return len(self.fields.data)

    def get_fields(self) -> list[IndexType]:
        return [i for i in self.fields.data]

    def __init__(self, shape: list[IndexType] | ArrayAttr[IndexType]) -> None:
        if isinstance(shape, list):
            shape = ArrayAttr(shape)
        super().__init__([shape])


@irdl_op_definition
class GetOp(IRDLOperation):
    name: str = "transfer.get"

    abs_val: Annotated[Operand, AbstractValueType]
    index: OpAttr[IntegerAttr[IndexType]]
    result: Annotated[OpResult, IndexType]

    def verify_(self) -> None:
        assert isinstance(self.abs_val.typ, AbstractValueType)
        if self.index.value.data >= self.abs_val.typ.get_num_fields():
            raise VerifyException("The required field is out of range")


@irdl_op_definition
class MakeOp(IRDLOperation):
    name: str = "transfer.make"

    arguments: Annotated[VarOperand, IndexType]
    result: Annotated[OpResult, AbstractValueType]

    def verify_(self) -> None:
        assert isinstance(self.results[0].typ, AbstractValueType)
        if len(self.operands) != self.results[0].typ.get_num_fields():
            raise VerifyException(
                "The number of given arguments doesn't match the abstract value"
            )


Transfer = Dialect([GetOp, MakeOp], [AbstractValueType])
