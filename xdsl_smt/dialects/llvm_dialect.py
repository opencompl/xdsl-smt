# pyright: reportConstantRedefinition=false
from __future__ import annotations

# Expose all base LLVM operations and attributes
from xdsl.dialects.llvm import (
    AddOp as AddOp,
    SubOp as SubOp,
    MulOp as MulOp,
    UDivOp as UDivOp,
    SDivOp as SDivOp,
    URemOp as URemOp,
    SRemOp as SRemOp,
    AndOp as AndOp,
    OrOp as OrOp,
    XOrOp as XOrOp,
    ShlOp as ShlOp,
    LShrOp as LShrOp,
    AShrOp as AShrOp,
    ExtractValueOp as ExtractValueOp,
    InsertValueOp as InsertValueOp,
    InlineAsmOp as InlineAsmOp,
    UndefOp as UndefOp,
    AllocaOp as AllocaOp,
    GEPOp as GEPOp,
    IntToPtrOp as IntToPtrOp,
    NullOp as NullOp,
    LoadOp as LoadOp,
    StoreOp as StoreOp,
    GlobalOp as GlobalOp,
    AddressOfOp as AddressOfOp,
    FuncOp as FuncOp,
    CallOp as CallOp,
    ReturnOp as ReturnOp,
    ConstantOp as ConstantOp,
    CallIntrinsicOp as CallIntrinsicOp,
    ZeroOp as ZeroOp,
    LLVMStructType as LLVMStructType,
    LLVMPointerType as LLVMPointerType,
    LLVMArrayType as LLVMArrayType,
    LLVMVoidType as LLVMVoidType,
    LLVMFunctionType as LLVMFunctionType,
    LinkageAttr as LinkageAttr,
    CallingConventionAttr as CallingConventionAttr,
    FastMathAttr as FastMathAttr,
    ICmpOp as ICmpOp,
)

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
from xdsl.dialects.builtin import IntegerType
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


llvm.LLVM._operations.append(SelectOp)  # type: ignore
LLVM: Dialect = llvm.LLVM
