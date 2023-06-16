from __future__ import annotations

from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntegerAttr,
    i1,
)
from typing import Annotated

from xdsl.ir import ParametrizedAttribute, Dialect, TypeAttribute, OpResult, Attribute

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
class TransIntegerType(ParametrizedAttribute, TypeAttribute):
    name = "transfer.integer"


@irdl_op_definition
class Constant(IRDLOperation):
    name = "transfer.constant"
    op: Annotated[Operand, TransIntegerType]
    result: Annotated[OpResult, TransIntegerType]
    value: OpAttr[IntegerAttr[IndexType]]


@irdl_op_definition
class NegOp(IRDLOperation):
    name = "transfer.neg"
    op: Annotated[Operand, TransIntegerType]
    result: Annotated[OpResult, TransIntegerType]


@irdl_op_definition
class AddOp(IRDLOperation):
    name = "transfer.add"
    lhs: Annotated[Operand, TransIntegerType]
    rhs: Annotated[Operand, TransIntegerType]
    result: Annotated[OpResult, TransIntegerType]


@irdl_op_definition
class SubOp(IRDLOperation):
    name = "transfer.sub"
    lhs: Annotated[Operand, TransIntegerType]
    rhs: Annotated[Operand, TransIntegerType]
    result: Annotated[OpResult, TransIntegerType]


@irdl_op_definition
class AndOp(IRDLOperation):
    name = "transfer.and"
    lhs: Annotated[Operand, TransIntegerType]
    rhs: Annotated[Operand, TransIntegerType]
    result: Annotated[OpResult, TransIntegerType]


@irdl_op_definition
class OrOp(IRDLOperation):
    name = "transfer.or"
    lhs: Annotated[Operand, TransIntegerType]
    rhs: Annotated[Operand, TransIntegerType]
    result: Annotated[OpResult, TransIntegerType]


@irdl_op_definition
class XorOp(IRDLOperation):
    name = "transfer.xor"
    lhs: Annotated[Operand, TransIntegerType]
    rhs: Annotated[Operand, TransIntegerType]
    result: Annotated[OpResult, TransIntegerType]


@irdl_op_definition
class CmpOp(IRDLOperation):
    name = "transfer.cmp"
    predicate: OpAttr[IntegerAttr[IndexType]]
    lhs: Annotated[Operand, TransIntegerType]
    rhs: Annotated[Operand, TransIntegerType]
    result: Annotated[OpResult, i1]


@irdl_attr_definition
class AbstractValueType(ParametrizedAttribute, TypeAttribute):
    name = "abs_value"
    fields: ParameterDef[ArrayAttr[Attribute]]

    def get_num_fields(self) -> int:
        return len(self.fields.data)

    def get_fields(self):
        return [i for i in self.fields.data]

    def __init__(self, shape: list[Attribute] | ArrayAttr[Attribute]) -> None:
        if isinstance(shape, list):
            shape = ArrayAttr(shape)
        super().__init__([shape])


@irdl_op_definition
class GetOp(IRDLOperation):
    name = "transfer.get"

    abs_val: Annotated[Operand, AbstractValueType]
    index: OpAttr[IntegerAttr[IndexType]]
    result: Annotated[OpResult, IndexType]

    def verify_(self) -> None:
        assert isinstance(self.abs_val.typ, AbstractValueType)
        if self.index.value.data >= self.abs_val.typ.get_num_fields():
            raise VerifyException("The required field is out of range")


@irdl_op_definition
class MakeOp(IRDLOperation):
    name = "transfer.make"

    arguments: Annotated[VarOperand, IndexType]
    result: Annotated[OpResult, AbstractValueType]

    def verify_(self) -> None:
        assert isinstance(self.results[0].typ, AbstractValueType)
        if len(self.operands) != self.results[0].typ.get_num_fields():
            raise VerifyException(
                "The number of given arguments doesn't match the abstract value"
            )


Transfer = Dialect(
    [Constant, CmpOp, AndOp, OrOp, XorOp, AddOp, SubOp, GetOp, MakeOp, NegOp],
    [TransIntegerType, AbstractValueType],
)
