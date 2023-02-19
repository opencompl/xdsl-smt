from __future__ import annotations

from io import IOBase
from typing import Annotated, TypeVar

from xdsl.dialects.builtin import IntAttr

from xdsl.ir import (Attribute, Dialect, OpResult, Operation,
                     ParametrizedAttribute, SSAValue)
from xdsl.irdl import (OpAttr, Operand, ParameterDef, irdl_op_definition,
                       irdl_attr_definition)
from xdsl.parser import BaseParser
from xdsl.printer import Printer

from traits.smt_printer import SMTConversionCtx, SMTLibOp, SMTLibSort, SimpleSMTLibOp
from traits.effects import Pure


@irdl_attr_definition
class BitVectorType(ParametrizedAttribute, SMTLibSort):
    name = "smt.bv.bv"
    width: ParameterDef[IntAttr]

    def print_sort_to_smtlib(self, stream: IOBase):
        print(f"(_ BitVec {self.width.data})", file=stream, end='')

    @staticmethod
    def from_int(value: int) -> BitVectorType:
        return BitVectorType([IntAttr(value)])

    @staticmethod
    def parse_parameters(parser: BaseParser) -> list[Attribute]:
        parser.parse_char("<")
        width = parser.parse_int_literal()
        parser.parse_char(">")
        return [IntAttr.build(width)]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<", self.width.data, ">")


@irdl_attr_definition
class BitVectorValue(ParametrizedAttribute):
    name = "smt.bv.bv_val"

    value: ParameterDef[IntAttr]
    width: ParameterDef[IntAttr]

    @staticmethod
    def from_int_value(value: int, width: int = 32) -> BitVectorValue:
        return BitVectorValue([IntAttr(value), IntAttr(width)])

    def get_type(self) -> BitVectorType:
        return BitVectorType.from_int(self.width.data)

    def verify(self) -> None:
        if not (0 <= self.value.data < 2**self.width.data):
            raise ValueError("BitVector value out of range")

    def as_smtlib_str(self) -> str:
        return f"(_ bv{self.value.data} {self.width.data})"

    @staticmethod
    def parse_parameters(parser: BaseParser) -> list[Attribute]:
        parser.parse_char("<")
        value = parser.parse_int_literal()
        parser.parse_char(":")
        width = parser.parse_int_literal()
        parser.parse_char(">")
        return [IntAttr.build(value), IntAttr.build(width)]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<", self.value.data, ": ", self.width.data, ">")


@irdl_op_definition
class ConstantOp(Operation, Pure, SMTLibOp):
    name = "smt.bv.constant"
    value: OpAttr[BitVectorValue]
    res: Annotated[OpResult, BitVectorType]

    @staticmethod
    def from_int_value(value: int, width: int) -> ConstantOp:
        bv_value = BitVectorValue.from_int_value(value, width)
        return ConstantOp.create(result_types=[bv_value.get_type()],
                                 attributes={"value": bv_value})

    @classmethod
    def parse(cls, result_types: list[Attribute],
              parser: BaseParser) -> ConstantOp:
        attr = parser.parse_attribute()
        if not isinstance(attr, BitVectorValue):
            raise ValueError("Expected a bitvector value")
        return ConstantOp.create(result_types=[BitVectorType([attr.width])],
                                 attributes={'value': attr})

    def print(self, printer: Printer) -> None:
        printer.print(" ", self.value)

    def print_expr_to_smtlib(self, stream: IOBase,
                             ctx: SMTConversionCtx) -> None:
        print(self.value.as_smtlib_str(), file=stream, end='')


_OpT = TypeVar("_OpT", bound="BinaryBVOp")


class BinaryBVOp(Operation, Pure):
    res: Annotated[OpResult, BitVectorType]
    lhs: Annotated[Operand, BitVectorType]
    rhs: Annotated[Operand, BitVectorType]

    @classmethod
    def parse(cls: type[_OpT], result_types: list[Attribute],
              parser: BaseParser) -> _OpT:
        lhs = parser.parse_operand()
        parser.parse_char(",")
        rhs = parser.parse_operand()
        return cls.build(result_types=[lhs.typ], operands=[lhs, rhs])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.lhs)
        printer.print(", ")
        printer.print_ssa_value(self.rhs)

    @classmethod
    def get(cls, lhs: SSAValue, rhs: SSAValue) -> BinaryBVOp:
        return cls.create(result_types=[lhs.typ], operands=[lhs, rhs])

    def verify_(self):
        if not (self.res.typ == self.lhs.typ == self.rhs.typ):
            raise ValueError("Operands must have same type")


@irdl_op_definition
class AddOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.add"

    def op_name(self) -> str:
        return "bvadd"


@irdl_op_definition
class OrOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.or"

    def op_name(self) -> str:
        return "bvor"


@irdl_op_definition
class SDivOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.sdiv"

    def op_name(self) -> str:
        return "bvsdiv"


SMTBitVectorDialect = Dialect([ConstantOp, AddOp, OrOp, SDivOp],
                              [BitVectorType, BitVectorValue])
