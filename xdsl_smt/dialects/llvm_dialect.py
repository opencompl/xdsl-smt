# pyright: reportWildcardImportFromLibrary=false
# pyright: reportConstantRedefinition=false
from __future__ import annotations

# We import all the LLVM operations here, so we only access them from here.
# Once we will have added all LLVM operations in xds, we can remove this file.
from xdsl.dialects.arith import ComparisonOperation, signlessIntegerLike
from xdsl.dialects.llvm import *
from xdsl.dialects import llvm
from xdsl.ir import Dialect

CMPI_COMPARISON_OPERATIONS = [
    "eq",
    "ne",
    "ugt",
    "uge",
    "ult",
    "ule",
    "sgt",
    "sge",
    "slt",
    "sle",
]


@irdl_op_definition
class ICmpOp(IRDLOperation, ComparisonOperation):
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
                "ugt": 2,
                "uge": 3,
                "ult": 4,
                "ule": 5,
                "sgt": 6,
                "sge": 7,
                "slt": 8,
                "sle": 9,
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

    fastmathFlags = prop_def(FastMathAttr)

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
