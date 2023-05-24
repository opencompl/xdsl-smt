from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, IndexType, IntegerType, ContainerOf, TypeAttribute, i1
from typing import Annotated

from xdsl.ir import (
    ParametrizedAttribute,
    Dialect,
)

from xdsl.irdl import (
    VarOperand,
    irdl_attr_definition,
    irdl_op_definition,
    ParameterDef,
    Operand,
    OpResult,
    IRDLOperation,
    AnyOf,
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

    def __init__(self, shape: list[IntegerType] | ArrayAttr[IndexType]) -> None:
        if isinstance(shape, list):
            shape = ArrayAttr(shape)
        super().__init__([shape])


signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))

@irdl_op_definition
class GetOp(IRDLOperation):
    name: str = "transfer.get"

    absVal: Annotated[Operand, AbstractValueType]
    index: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, ContainerOf(IndexType)]

    def verify_(self) -> None:
        index = self.operands[1].op.value.value.data
        if index >= self.absVal.typ.get_num_fields():
            raise VerifyException("The required field is out of range")


@irdl_op_definition
class MakeOp(IRDLOperation):
    name: str = "transfer.make"

    arguments: Annotated[VarOperand, signlessIntegerLike]
    result: Annotated[OpResult, ContainerOf(AbstractValueType)]

    def verify_(self) -> None:
        assert isinstance(self.results[0].typ, AbstractValueType)
        if len(self.operands) != self.results[0].typ.get_num_fields():
            raise VerifyException("The number of given arguments doesn't match the abstract value")


Transfer = Dialect(
    [GetOp, MakeOp],
    [AbstractValueType]
)
