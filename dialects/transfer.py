from __future__ import annotations
from abc import ABC
from z3 import *

from dataclasses import dataclass, field
from enum import Enum
from xdsl.dialects.builtin import AnyIntegerAttr
from xdsl.dialects.builtin import ArrayAttr
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.builtin import IndexType
from xdsl.dialects.builtin import IntegerType
from xdsl.dialects.builtin import ContainerOf
from xdsl.dialects.builtin import TypeAttribute
from xdsl.dialects.builtin import i1
from xdsl.dialects.arith import Constant
import functools
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, List, Sequence, TypeVar, Union, Set, Optional

from xdsl.ir import (
    ParametrizedAttribute,
    Dialect,
)

from xdsl.irdl import (
    AllOf,
    OpAttr,
    VarOpResult,
    VarOperand,
    VarRegion,
    irdl_attr_definition,
    attr_constr_coercion,
    irdl_data_definition,
    irdl_to_attr_constraint,
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
    name: str = "absValue"

    fields: ParameterDef[ArrayAttr[IndexType]]

    def get_num_fields(self) -> int:
        return len(self.fields.data)

    def get_fields(self) -> list[IndexType]:
        return [i for i in self.fields.data]

    def __init__(self, shape: list[IntegerType] | ArrayAttr[IndexType]) -> None:
        if isinstance(shape, list):
            shape = ArrayAttr(shape)
        super().__init__([shape])


'''
    @staticmethod
    def from_shape(shape: Sequence[int | IntegerAttr[IndexType]]):
        return AbstractValueType(
            ArrayAttr(
                    [
                        IntegerType.from_width(d)
                        if isinstance(d, int)
                        else d
                        for d in shape
                    ]))
'''

signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))


@irdl_op_definition
class IfOp(IRDLOperation):
    def verify_(self) -> None:
        if len(self.operands) != 3 or len(self.results) != 1:
            raise VerifyException("IF operation expects 3 operands and 1 result.")
        if not (self.operands[0].typ == i1):
            raise VerifyException("IF operand 0 has to be i1 type")
        if not (self.operands[1].typ == self.operands[2].typ == self.results[0].typ):
            raise VerifyException("expect all input and result types to be equal")

    name: str = "transfer.if"

    cond: Annotated[Operand, ContainerOf(i1)]
    lhs: Annotated[Operand, IndexType]
    rhs: Annotated[Operand, IndexType]
    result: Annotated[OpResult, IndexType]


@irdl_op_definition
class NegOp(IRDLOperation):
    def verify_(self) -> None:
        if len(self.operands) != 1 or len(self.results) != 1:
            raise VerifyException("NEG operation expects 1 operands and 1 result.")

    name: str = "transfer.neg"
    val: Annotated[Operand, IndexType]
    result: Annotated[Operand, IndexType]


@irdl_op_definition
class GetOp(IRDLOperation):
    name: str = "transfer.get"

    absVal: Annotated[Operand, ContainerOf(AbstractValueType)]
    index: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, ContainerOf(IndexType)]

    def verify_(self) -> None:
        if len(self.operands) != 2 or len(self.results) != 1:
            raise VerifyException("GET operation expects 2 operands and 1 result.")
        if not isinstance(self.operands[1].op, Constant):
            raise VerifyException("The second operand has to be arith.constant!")
        index = self.operands[1].op.value.value.data
        assert isinstance(self.absVal.typ, AbstractValueType)
        if index >= self.absVal.typ.get_num_fields():
            raise VerifyException("The required field is out of range")
        '''
        fields=self.absVal.typ.get_fields()
        if self.results[0].typ!=fields[index]:
            raise VerifyException("The type of returned value doesn't match accessed value")
        '''


@irdl_op_definition
class MakeOp(IRDLOperation):
    name: str = "transfer.make"

    arguments: Annotated[VarOperand, signlessIntegerLike]
    result: Annotated[OpResult, ContainerOf(AbstractValueType)]

    def verify_(self) -> None:
        assert isinstance(self.results[0].typ, AbstractValueType)
        if len(self.operands) != self.results[0].typ.get_num_fields():
            raise VerifyException("The number of given arguments doesn't match the abstract value")
        '''
        fields=self.results[0].typ.get_fields()
        for i in range(len(fields)):
            if self.operands[i].typ != fields[i]:
                raise VerifyException("The "+str(i+1)+"th argument doesn't match the abstract value")
        '''


Transfer = Dialect(
    [IfOp, GetOp, MakeOp, NegOp],
    [AbstractValueType]
)
