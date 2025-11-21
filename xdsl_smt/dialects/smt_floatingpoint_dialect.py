from __future__ import annotations
from abc import abstractmethod

from xdsl.printer import Printer

from xdsl.parser import AttrParser

from xdsl.dialects.builtin import (
    IntAttr,
)
from typing import IO, Sequence
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType

from xdsl.ir import (
    ParametrizedAttribute,
    TypeAttribute,
    OpResult,
    Attribute,
    Operation,
    Dialect,
)

from xdsl.irdl import (
    operand_def,
    result_def,
    Operand,
    irdl_attr_definition,
    irdl_op_definition,
    IRDLOperation,
)
from xdsl.utils.exceptions import VerifyException
from ..traits.effects import Pure

from ..traits.smt_printer import (
    SMTLibOp,
    SimpleSMTLibOp,
    SMTLibSort,
    SMTConversionCtx,
)


@irdl_attr_definition
class FloatingPointType(ParametrizedAttribute, SMTLibSort, TypeAttribute):
    """
    eb defines the number of bits in the exponent;
    sb defines the number of bits in the significand, *including * the hidden bit.
    """

    name = "smt.fp"
    eb: IntAttr
    sb: IntAttr

    def __init__(self, eb: int | IntAttr, sb: int | IntAttr):
        if isinstance(eb, int):
            eb = IntAttr(eb)
        if isinstance(sb, int):
            sb = IntAttr(sb)
        super().__init__(eb, sb)

    def verify(self) -> None:
        super().verify()
        if self.eb.data <= 0:
            raise VerifyException(
                "FloatingPointType exponent must be strictly greater "
                f"than zero, got {self.eb.data}"
            )
        if self.sb.data <= 0:
            raise VerifyException(
                "FloatingPointType significand must be strictly greater "
                f"than zero, got {self.sb.data}"
            )

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            eb = parser.parse_integer(allow_boolean=False, allow_negative=False)
            parser.parse_characters(",")
            sb = parser.parse_integer(allow_boolean=False, allow_negative=False)

        return IntAttr(eb), IntAttr(sb)

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string(f"<{self.eb.data}, {self.sb.data}>")

    def print_sort_to_smtlib(self, stream: IO[str]) -> None:
        print(f"(_ FloatingPoint {self.eb.data} {self.sb.data})", file=stream, end="")


"""
These correspond to the IEEE binary16, binary32, binary64 and binary128 formats.
"""
float16 = FloatingPointType(5, 11)
float32 = FloatingPointType(8, 24)
float64 = FloatingPointType(11, 53)
float128 = FloatingPointType(15, 113)


def getWidthFromBitVectorType(typ: Attribute):
    if isinstance(typ, BitVectorType):
        return typ.width.data
    raise ValueError("Expected a BitVector type")


################################################################################
#                          FP Value Constructors                               #
################################################################################


@irdl_op_definition
class ConstantOp(IRDLOperation, Pure, SimpleSMTLibOp):
    """
    FP literals as bit string triples, with the leading bit for the significand not represented (hidden bit)
    (fp (_ BitVec 1) (_ BitVec eb) (_ BitVec i) (_ FloatingPoint eb sb))
    where eb and sb are numerals greater than 1 and i = sb - 1.
    """

    name = "smt.fp.constant"
    lsb = operand_def(BitVectorType)
    eb = operand_def(BitVectorType)
    rsb = operand_def(BitVectorType)
    result = result_def(FloatingPointType)

    def __init__(self, lsb: Operand, eb: Operand, rsb: Operand):
        eb_len = getWidthFromBitVectorType(eb.type)
        sb_len = getWidthFromBitVectorType(lsb.type) + getWidthFromBitVectorType(
            rsb.type
        )
        super().__init__(
            result_types=[FloatingPointType(eb_len, sb_len)], operands=[lsb, eb, rsb]
        )

    def verify_(self):
        if not (1 == getWidthFromBitVectorType(self.lsb.type)):
            raise VerifyException("Expected leading significant bit with width 1")

    def op_name(self) -> str:
        return "fp"


class SpecialConstantOp(IRDLOperation, Pure, SMTLibOp):
    """
    This class is an abstract class for -/+infinity, -/+zero and NaN
    """

    res: OpResult = result_def(FloatingPointType)

    def __init__(self, eb: int | IntAttr, sb: int | IntAttr):
        super().__init__(result_types=[FloatingPointType(eb, sb)])

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx) -> None:
        assert isinstance(self, Operation)
        assert isinstance(self.res.type, FloatingPointType)
        print(f"(_ {self.constant_name()}", file=stream, end="")
        print(f" {self.res.type.eb.data} {self.res.type.sb.data})", file=stream, end="")

    @abstractmethod
    def constant_name(self) -> str:
        """Expression name when printed in SMTLib."""
        ...


@irdl_op_definition
class PositiveInfinityOp(SpecialConstantOp):
    name = "smt.fp.pinf"

    def constant_name(self) -> str:
        return "+oo"


@irdl_op_definition
class NegativeInfinityOp(SpecialConstantOp):
    name = "smt.fp.ninf"

    def constant_name(self) -> str:
        return "-oo"


@irdl_op_definition
class PositiveZeroOp(SpecialConstantOp):
    name = "smt.fp.pzero"

    def constant_name(self) -> str:
        return "+zero"


@irdl_op_definition
class NegativeZeroOp(SpecialConstantOp):
    name = "smt.fp.nzero"

    def constant_name(self) -> str:
        return "-zero"


@irdl_op_definition
class NaNOp(SpecialConstantOp):
    name = "smt.fp.nan"

    def constant_name(self) -> str:
        return "nan"


SMTFloatingPointDialect = Dialect(
    "smt.fp",
    [
        ConstantOp,
        PositiveZeroOp,
        NegativeZeroOp,
        PositiveInfinityOp,
        NegativeInfinityOp,
        NaNOp,
    ],
    [FloatingPointType],
)
