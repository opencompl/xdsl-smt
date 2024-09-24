# pyright: reportConstantRedefinition=false
from __future__ import annotations

# Expose all base LLVM operations and attributes
# pyright: reportUnusedImport=false
from xdsl.dialects.llvm import (
    AddOp,
    SubOp,
    MulOp,
    UDivOp,
    SDivOp,
    URemOp,
    SRemOp,
    AndOp,
    OrOp,
    XOrOp,
    ShlOp,
    LShrOp,
    AShrOp,
    ExtractValueOp,
    InsertValueOp,
    InlineAsmOp,
    UndefOp,
    AllocaOp,
    GEPOp,
    IntToPtrOp,
    NullOp,
    LoadOp,
    StoreOp,
    GlobalOp,
    AddressOfOp,
    FuncOp,
    CallOp,
    ReturnOp,
    ConstantOp,
    CallIntrinsicOp,
    ZeroOp,
    LLVMStructType,
    LLVMPointerType,
    LLVMArrayType,
    LLVMVoidType,
    LLVMFunctionType,
    LinkageAttr,
    CallingConventionAttr,
    FastMathAttr,
)
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
