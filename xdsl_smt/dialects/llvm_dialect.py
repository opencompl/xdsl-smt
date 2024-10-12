# pyright: reportConstantRedefinition=false
from __future__ import annotations

# Expose all base LLVM operations and attributes
# pyright: reportUnusedImport=false
from xdsl.dialects.llvm import AddOp as AddOp
from xdsl.dialects.llvm import SubOp as SubOp
from xdsl.dialects.llvm import MulOp as MulOp
from xdsl.dialects.llvm import UDivOp as UDivOp
from xdsl.dialects.llvm import SDivOp as SDivOp
from xdsl.dialects.llvm import URemOp as URemOp
from xdsl.dialects.llvm import SRemOp as SRemOp
from xdsl.dialects.llvm import AndOp as AndOp
from xdsl.dialects.llvm import OrOp as OrOp
from xdsl.dialects.llvm import XOrOp as XOrOp
from xdsl.dialects.llvm import ShlOp as ShlOp
from xdsl.dialects.llvm import LShrOp as LShrOp
from xdsl.dialects.llvm import AShrOp as AShrOp
from xdsl.dialects.llvm import ExtractValueOp as ExtractValueOp
from xdsl.dialects.llvm import InsertValueOp as InsertValueOp
from xdsl.dialects.llvm import InlineAsmOp as InlineAsmOp
from xdsl.dialects.llvm import UndefOp as UndefOp
from xdsl.dialects.llvm import AllocaOp as AllocaOp
from xdsl.dialects.llvm import GEPOp as GEPOp
from xdsl.dialects.llvm import IntToPtrOp as IntToPtrOp
from xdsl.dialects.llvm import NullOp as NullOp
from xdsl.dialects.llvm import LoadOp as LoadOp
from xdsl.dialects.llvm import StoreOp as StoreOp
from xdsl.dialects.llvm import GlobalOp as GlobalOp
from xdsl.dialects.llvm import AddressOfOp as AddressOfOp
from xdsl.dialects.llvm import FuncOp as FuncOp
from xdsl.dialects.llvm import CallOp as CallOp
from xdsl.dialects.llvm import ReturnOp as ReturnOp
from xdsl.dialects.llvm import ConstantOp as ConstantOp
from xdsl.dialects.llvm import CallIntrinsicOp as CallIntrinsicOp
from xdsl.dialects.llvm import ZeroOp as ZeroOp
from xdsl.dialects.llvm import LLVMStructType as LLVMStructType
from xdsl.dialects.llvm import LLVMPointerType as LLVMPointerType
from xdsl.dialects.llvm import LLVMArrayType as LLVMArrayType
from xdsl.dialects.llvm import LLVMVoidType as LLVMVoidType
from xdsl.dialects.llvm import LLVMFunctionType as LLVMFunctionType
from xdsl.dialects.llvm import LinkageAttr as LinkageAttr
from xdsl.dialects.llvm import CallingConventionAttr as CallingConventionAttr
from xdsl.dialects.llvm import FastMathAttr as FastMathAttr

from xdsl.dialects.arith import ComparisonOperation, signlessIntegerLike
from xdsl.dialects import llvm
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.ir import Dialect, OpResult, Operation, SSAValue, Attribute
from xdsl.irdl import (
    irdl_op_definition,
    Operand,
    operand_def,
    prop_def,
    result_def,
    IRDLOperation,
)
from xdsl.dialects.builtin import AnyIntegerAttr, IntegerType, IntegerAttr
from xdsl.utils.exceptions import VerifyException

CMPI_COMPARISON_OPERATIONS = [
    "eq",
    "ne",
    "slt",
    "sle",
    "sgt",
    "sge",
    "ult",
    "ule",
    "ugt",
    "uge",
]


@irdl_op_definition
class ICmpOp(ComparisonOperation):
    name = "llvm.icmp"
    predicate: AnyIntegerAttr = prop_def(AnyIntegerAttr)
    lhs: Operand = operand_def(signlessIntegerLike)
    rhs: Operand = operand_def(signlessIntegerLike)
    result: OpResult = result_def(IntegerType(1))

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        arg: int | str,
    ):
        operand1 = SSAValue.get(operand1)
        operand2 = SSAValue.get(operand2)
        ICmpOp._validate_operand_types(operand1, operand2)

        if isinstance(arg, str):
            cmpi_comparison_operations = {
                "eq": 0,
                "ne": 1,
                "slt": 2,
                "sle": 3,
                "sgt": 4,
                "sge": 5,
                "ult": 6,
                "ule": 7,
                "ugt": 8,
                "uge": 9,
            }
            arg = ICmpOp._get_comparison_predicate(arg, cmpi_comparison_operations)

        return super().__init__(
            operands=[operand1, operand2],
            result_types=[IntegerType(1)],
            properties={"predicate": IntegerAttr.from_int_and_width(arg, 64)},
        )

    @classmethod
    def parse(cls, parser: Parser):
        arg = parser.parse_identifier()
        parser.parse_punctuation(",")
        operand1 = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        operand2 = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        input_type = parser.parse_type()
        (operand1, operand2) = parser.resolve_operands(
            [operand1, operand2], 2 * [input_type], parser.pos
        )

        return cls(operand1, operand2, arg)

    def print(self, printer: Printer):
        printer.print(" ")

        printer.print_string(CMPI_COMPARISON_OPERATIONS[self.predicate.value.data])
        printer.print(", ")
        printer.print_operand(self.lhs)
        printer.print(", ")
        printer.print_operand(self.rhs)
        printer.print(" : ")
        printer.print_attribute(self.lhs.type)


@irdl_op_definition
class SelectOp(IRDLOperation):
    name = "llvm.select"
    cond: Operand = operand_def(IntegerType(1))  # should be unsigned
    lhs: Operand = operand_def(Attribute)
    rhs: Operand = operand_def(Attribute)
    result: OpResult = result_def(Attribute)

    fastmathFlags = prop_def(llvm.FastMathAttr)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.cond.type != IntegerType(1):
            raise VerifyException("Condition has to be of type !i1")
        if self.lhs.type != self.rhs.type or self.rhs.type != self.result.type:
            raise VerifyException("expect all input and output types to be equal")

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
        operand3: Operation | SSAValue,
    ):
        operand2 = SSAValue.get(operand2)
        return super().__init__(
            operands=[operand1, operand2, operand3], result_types=[operand2.type]
        )

    @classmethod
    def parse(cls, parser: Parser):
        cond = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        operand1 = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        operand2 = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        result_type = parser.parse_type()
        (cond, operand1, operand2) = parser.resolve_operands(
            [cond, operand1, operand2],
            [IntegerType(1), result_type, result_type],
            parser.pos,
        )

        return cls(cond, operand1, operand2)

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_operand(self.cond)
        printer.print(", ")
        printer.print_operand(self.lhs)
        printer.print(", ")
        printer.print_operand(self.rhs)
        printer.print(" : ")
        printer.print_attribute(self.result.type)


llvm.LLVM._operations.append(ICmpOp)  # type: ignore
llvm.LLVM._operations.append(SelectOp)  # type: ignore
LLVM: Dialect = llvm.LLVM
