from __future__ import annotations
from abc import ABC

from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    i1,
)
from typing import Annotated, Mapping, Sequence

from xdsl.ir import (
    ParametrizedAttribute,
    Dialect,
    TypeAttribute,
    OpResult,
    Attribute,
)
from xdsl.utils.hints import isa

from xdsl.irdl import (
    ConstraintVar,
    attr_def,
    operand_def,
    var_operand_def,
    result_def,
    Operand,
    VarOperand,
    irdl_attr_definition,
    irdl_op_definition,
    ParameterDef,
    IRDLOperation,
)
from xdsl.utils.exceptions import VerifyException

from ..traits.infer_type import InferResultTypeInterface


@irdl_attr_definition
class TransIntegerType(ParametrizedAttribute, TypeAttribute):
    name = "transfer.integer"


@irdl_op_definition
class Constant(IRDLOperation, InferResultTypeInterface):
    name = "transfer.constant"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    op: Operand = operand_def(T)
    result: OpResult = result_def(T)
    value: IntegerAttr[IndexType] = attr_def(IntegerAttr[IndexType])

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [op]:
                return [op]
            case _:
                raise VerifyException("Constant operation expects exactly one operand")


class UnaryOp(IRDLOperation, InferResultTypeInterface, ABC):
    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    op: Operand = operand_def(T)
    result: OpResult = result_def(T)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [op]:
                return [op]
            case _:
                raise VerifyException("Unary operation expects exactly one operand")


@irdl_op_definition
class NegOp(UnaryOp):
    name = "transfer.neg"


class BinOp(IRDLOperation, InferResultTypeInterface, ABC):
    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(T)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [lhs, _]:
                return [lhs]
            case _:
                raise VerifyException("Bin operation expects exactly two operands")


class PredicateOp(IRDLOperation, InferResultTypeInterface, ABC):
    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(i1)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [_, _]:
                return [i1]
            case _:
                raise VerifyException("Bin operation expects exactly two operands")


@irdl_op_definition
class AddOp(BinOp):
    name = "transfer.add"


@irdl_op_definition
class SubOp(BinOp):
    name = "transfer.sub"


@irdl_op_definition
class MulOp(BinOp):
    name = "transfer.mul"


@irdl_op_definition
class UMulOverflowOp(PredicateOp):
    name = "transfer.umul_overflow"


@irdl_op_definition
class AndOp(BinOp):
    name = "transfer.and"


@irdl_op_definition
class OrOp(BinOp):
    name = "transfer.or"


@irdl_op_definition
class XorOp(BinOp):
    name = "transfer.xor"


@irdl_op_definition
class GetBitWidthOp(UnaryOp):
    name = "transfer.get_bit_width"


@irdl_op_definition
class CountLZeroOp(UnaryOp):
    name = "transfer.countl_zero"


@irdl_op_definition
class CountRZeroOp(UnaryOp):
    name = "transfer.countr_zero"


@irdl_op_definition
class CountLOneOp(UnaryOp):
    name = "transfer.countl_one"


@irdl_op_definition
class CountROneOp(UnaryOp):
    name = "transfer.countr_one"


@irdl_op_definition
class SMinOp(BinOp):
    name = "transfer.smin"


@irdl_op_definition
class SMaxOp(BinOp):
    name = "transfer.smax"


@irdl_op_definition
class UMinOp(BinOp):
    name = "transfer.umin"


@irdl_op_definition
class UMaxOp(BinOp):
    name = "transfer.umax"


@irdl_op_definition
class GetLowBitsOp(IRDLOperation):
    name = "transfer.get_low_bits"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    val: Operand = operand_def(T)
    low_bits: Operand = operand_def(T)
    result: OpResult = result_def(T)


@irdl_op_definition
class SetHighBitsOp(IRDLOperation):
    name = "transfer.set_high_bits"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    val: Operand = operand_def(T)
    high_bits: Operand = operand_def(T)
    result: OpResult = result_def(T)


@irdl_op_definition
class CmpOp(PredicateOp):
    name = "transfer.cmp"

    predicate: IntegerAttr[IndexType] = attr_def(IntegerAttr[IndexType])


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
class GetOp(IRDLOperation, InferResultTypeInterface):
    name = "transfer.get"

    abs_val: Operand = operand_def(AbstractValueType)
    index: IntegerAttr[IndexType] = attr_def(IntegerAttr[IndexType])
    result: OpResult = result_def(Attribute)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        if len(operand_types) != 1 or not isinstance(
            operand_types[0], AbstractValueType
        ):
            raise VerifyException("Get operation expects exactly one abs_value operand")
        if "index" not in attributes:
            raise VerifyException("Get operation expects an index attribute")
        if not isa(attributes["index"], IntegerAttr[IndexType]):
            raise VerifyException("Get operation expects an integer index attribute")
        if attributes["index"].value.data >= operand_types[0].get_num_fields():
            raise VerifyException("'index' attribute is out of range")
        return [operand_types[0].get_fields()[attributes["index"].value.data]]

    def verify_(self) -> None:
        if self.infer_result_type(
            [operand.type for operand in self.operands], self.attributes
        ) != [self.result.type]:
            raise VerifyException("The result type doesn't match the inferred type")


@irdl_op_definition
class MakeOp(IRDLOperation, InferResultTypeInterface):
    name = "transfer.make"

    arguments: VarOperand = var_operand_def(Attribute)
    result: OpResult = result_def(AbstractValueType)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        return [AbstractValueType(list(operand_types))]

    def verify_(self) -> None:
        assert isinstance(self.result.type, AbstractValueType)
        if len(self.operands) != self.result.type.get_num_fields():
            raise VerifyException(
                "The number of given arguments doesn't match the abstract value"
            )
        if self.result.type.get_fields() != [arg.type for arg in self.arguments]:
            raise VerifyException("The required field doesn't match the result type")


Transfer = Dialect(
    [
        Constant,
        CmpOp,
        AndOp,
        OrOp,
        XorOp,
        AddOp,
        SubOp,
        GetOp,
        MakeOp,
        NegOp,
        MulOp,
        CountLOneOp,
        CountLZeroOp,
        CountROneOp,
        CountRZeroOp,
        SetHighBitsOp,
        GetLowBitsOp,
        GetBitWidthOp,
        SMinOp,
        SMaxOp,
        UMaxOp,
        UMinOp,
        UMulOverflowOp,
    ],
    [TransIntegerType, AbstractValueType],
)
