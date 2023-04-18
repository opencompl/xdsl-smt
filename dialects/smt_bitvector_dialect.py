from __future__ import annotations

from typing import Annotated, TypeVar, IO

from xdsl.dialects.builtin import IntAttr

from xdsl.ir import (Attribute, Dialect, OpResult, ParametrizedAttribute,
                     SSAValue)
from xdsl.irdl import (OpAttr, Operand, ParameterDef, irdl_op_definition,
                       irdl_attr_definition, IRDLOperation)
from xdsl.parser import BaseParser
from xdsl.printer import Printer

from traits.smt_printer import SMTConversionCtx, SMTLibOp, SMTLibSort, SimpleSMTLibOp
from traits.effects import Pure
from dialects.smt_dialect import BoolType


@irdl_attr_definition
class BitVectorType(ParametrizedAttribute, SMTLibSort):
    name = "smt.bv.bv"
    width: ParameterDef[IntAttr]

    def print_sort_to_smtlib(self, stream: IO[str]):
        print(f"(_ BitVec {self.width.data})", file=stream, end='')

    def __init__(self, value: int | IntAttr):
        if isinstance(value, int):
            value = IntAttr(value)
        super().__init__([value])

    @staticmethod
    def from_int(value: int) -> BitVectorType:
        return BitVectorType(value)

    @staticmethod
    def parse_parameters(parser: BaseParser) -> list[Attribute]:
        parser.parse_char("<")
        width = parser.parse_int_literal()
        parser.parse_char(">")
        return [IntAttr(width)]

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
        return [IntAttr(value), IntAttr(width)]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<", self.value.data, ": ", self.width.data, ">")


@irdl_op_definition
class ConstantOp(IRDLOperation, Pure, SMTLibOp):
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
        return ConstantOp.create(result_types=[BitVectorType(attr.width)],
                                 attributes={'value': attr})

    def print(self, printer: Printer) -> None:
        printer.print(" ", self.value)

    def print_expr_to_smtlib(self, stream: IO[str],
                             ctx: SMTConversionCtx) -> None:
        print(self.value.as_smtlib_str(), file=stream, end='')


_UOpT = TypeVar("_UOpT", bound="UnaryBVOp")


class UnaryBVOp(IRDLOperation, Pure):
    res: Annotated[OpResult, BitVectorType]
    arg: Annotated[Operand, BitVectorType]

    @classmethod
    def parse(cls: type[_UOpT], result_types: list[Attribute],
              parser: BaseParser) -> _UOpT:
        arg = parser.parse_operand()
        return cls.build(result_types=[arg.typ], operands=[arg])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.arg)

    @classmethod
    def get(cls: type[_UOpT], arg: SSAValue) -> _UOpT:
        return cls.create(result_types=[arg.typ], operands=[arg])

    def verify_(self):
        if not (self.res.typ == self.arg.typ):
            raise ValueError("Operand and result must have the same type")


_BOpT = TypeVar("_BOpT", bound="BinaryBVOp")


class BinaryBVOp(IRDLOperation, Pure):
    res: Annotated[OpResult, BitVectorType]
    lhs: Annotated[Operand, BitVectorType]
    rhs: Annotated[Operand, BitVectorType]

    @classmethod
    def parse(cls: type[_BOpT], result_types: list[Attribute],
              parser: BaseParser) -> _BOpT:
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
    def get(cls: type[_BOpT], lhs: SSAValue, rhs: SSAValue) -> _BOpT:
        return cls.create(result_types=[lhs.typ], operands=[lhs, rhs])

    def verify_(self):
        if not (self.res.typ == self.lhs.typ == self.rhs.typ):
            raise ValueError("Operands must have same type")


################################################################################
#                          Basic Bitvector Arithmetic                          #
################################################################################


@irdl_op_definition
class AddOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.add"

    def op_name(self) -> str:
        return "bvadd"


@irdl_op_definition
class SubOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.sub"

    def op_name(self) -> str:
        return "bvsub"


@irdl_op_definition
class NegOp(UnaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.neg"

    def op_name(self) -> str:
        return "bvneg"


@irdl_op_definition
class MulOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.mul"

    def op_name(self) -> str:
        return "bvmul"


@irdl_op_definition
class URemOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.urem"

    def op_name(self) -> str:
        return "bvurem"


@irdl_op_definition
class SRemOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.srem"

    def op_name(self) -> str:
        return "bvsrem"


@irdl_op_definition
class SModOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.smod"

    def op_name(self) -> str:
        return "bvsmod"


@irdl_op_definition
class ShlOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.shl"

    def op_name(self) -> str:
        return "bvshl"


@irdl_op_definition
class LShrOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.lshr"

    def op_name(self) -> str:
        return "bvlshr"


@irdl_op_definition
class AShrOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.ashr"

    def op_name(self) -> str:
        return "bvashr"


@irdl_op_definition
class SDivOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.sdiv"

    def op_name(self) -> str:
        return "bvsdiv"


@irdl_op_definition
class UDivOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.udiv"

    def op_name(self) -> str:
        return "bvudiv"


################################################################################
#                                   Bitwise                                    #
################################################################################


@irdl_op_definition
class OrOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.or"

    def op_name(self) -> str:
        return "bvor"


@irdl_op_definition
class AndOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.and"

    def op_name(self) -> str:
        return "bvand"


@irdl_op_definition
class NotOp(UnaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.not"

    def op_name(self) -> str:
        return "bvnot"


@irdl_op_definition
class NAndOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.nand"

    def op_name(self) -> str:
        return "bvnand"


@irdl_op_definition
class NorOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.nor"

    def op_name(self) -> str:
        return "bvnor"


@irdl_op_definition
class XNorOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.xnor"

    def op_name(self) -> str:
        return "bvxnor"


################################################################################
#                                  Predicate                                   #
################################################################################

_BPOpT = TypeVar("_BPOpT", bound="BinaryPredBVOp")


class BinaryPredBVOp(IRDLOperation, Pure):
    res: Annotated[OpResult, BoolType]
    lhs: Annotated[Operand, BitVectorType]
    rhs: Annotated[Operand, BitVectorType]

    @classmethod
    def parse(cls: type[_BPOpT], result_types: list[Attribute],
              parser: BaseParser) -> _BPOpT:
        lhs = parser.parse_operand()
        parser.parse_char(",")
        rhs = parser.parse_operand()
        return cls.get(lhs, rhs)

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.lhs)
        printer.print(", ")
        printer.print_ssa_value(self.rhs)

    @classmethod
    def get(cls: type[_BPOpT], lhs: SSAValue, rhs: SSAValue) -> _BPOpT:
        return cls.create(result_types=[BoolType()], operands=[lhs, rhs])

    def verify_(self):
        if not (self.lhs.typ == self.rhs.typ):
            raise ValueError("Operands must have the same type")


@irdl_op_definition
class UleOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.ule"

    def op_name(self) -> str:
        return "bvule"


@irdl_op_definition
class UltOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.ult"

    def op_name(self) -> str:
        return "bvult"


@irdl_op_definition
class UgeOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.uge"

    def op_name(self) -> str:
        return "bvuge"


@irdl_op_definition
class UgtOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.ugt"

    def op_name(self) -> str:
        return "bvugt"


@irdl_op_definition
class SleOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.sle"

    def op_name(self) -> str:
        return "bvsle"


@irdl_op_definition
class SltOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.slt"

    def op_name(self) -> str:
        return "bvslt"


@irdl_op_definition
class SgeOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.sge"

    def op_name(self) -> str:
        return "bvsge"


@irdl_op_definition
class SgtOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.sgt"

    def op_name(self) -> str:
        return "bvsgt"


SMTBitVectorDialect = Dialect(
    [
        ConstantOp,
        # Arithmetic
        NegOp,
        AddOp,
        SubOp,
        MulOp,
        URemOp,
        SRemOp,
        SModOp,
        ShlOp,
        LShrOp,
        AShrOp,
        UDivOp,
        SDivOp,
        # Bitwise
        NotOp,
        OrOp,
        AndOp,
        NAndOp,
        NorOp,
        XNorOp,
        # Predicate
        UleOp,
        UltOp,
        UgeOp,
        UgtOp,
        SleOp,
        SltOp,
        SgeOp,
        SgtOp,
    ],
    [BitVectorType, BitVectorValue])
