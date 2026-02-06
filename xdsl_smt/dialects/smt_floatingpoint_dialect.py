from __future__ import annotations
from abc import abstractmethod

from xdsl.printer import Printer

from xdsl.parser import AttrParser

from xdsl.dialects.builtin import (
    IntAttr,
)
from typing import IO, Sequence
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType
from xdsl_smt.dialects.smt_dialect import BoolType

from xdsl.ir import (
    ParametrizedAttribute,
    TypeAttribute,
    OpResult,
    Attribute,
    Operation,
    Dialect,
    SSAValue,
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
class RoundingModeType(ParametrizedAttribute, SMTLibSort, TypeAttribute):
    """
    Defines Rounding Mode of FP operations, it includes following constants and their abbreviated version
    :funs ((roundNearestTiesToEven RoundingMode) (RNE RoundingMode)
        (roundNearestTiesToAway RoundingMode) (RNA RoundingMode)
        (roundTowardPositive RoundingMode)    (RTP RoundingMode)
        (roundTowardNegative RoundingMode)    (RTN RoundingMode)
        (roundTowardZero RoundingMode)        (RTZ RoundingMode)
        )
    """

    name = "smt.fp.rounding_mode"

    def __init__(self):
        super().__init__()

    def print_sort_to_smtlib(self, stream: IO[str]) -> None:
        print(f"RoundingMode", file=stream, end="")


class RunningModeConstantOp(IRDLOperation, Pure, SMTLibOp):
    """
    This class is an abstract class for all RoundingMode constants
    :funs ((roundNearestTiesToEven RoundingMode) (RNE RoundingMode)
        (roundNearestTiesToAway RoundingMode) (RNA RoundingMode)
        (roundTowardPositive RoundingMode)    (RTP RoundingMode)
        (roundTowardNegative RoundingMode)    (RTN RoundingMode)
        (roundTowardZero RoundingMode)        (RTZ RoundingMode)
        )
    """

    res: OpResult = result_def(RoundingModeType)

    def __init__(self):
        super().__init__(result_types=[RoundingModeType()])

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx) -> None:
        print(f"{self.constant_name()}", file=stream, end="")

    @abstractmethod
    def constant_name(self) -> str:
        """RoundingMode name when printed in SMTLib."""
        ...


@irdl_op_definition
class RoundNearestTiesToEvenOp(RunningModeConstantOp):
    name = "smt.fp.round_nearest_ties_to_even"

    def constant_name(self) -> str:
        return "roundNearestTiesToEven"


@irdl_op_definition
class RNEOp(RunningModeConstantOp):
    name = "smt.fp.rne"

    def constant_name(self) -> str:
        return "RNE"


@irdl_op_definition
class RoundNearestTiesToAwayOp(RunningModeConstantOp):
    name = "smt.fp.round_nearest_ties_to_away"

    def constant_name(self) -> str:
        return "roundNearestTiesToAway"


@irdl_op_definition
class RNAOp(RunningModeConstantOp):
    name = "smt.fp.rna"

    def constant_name(self) -> str:
        return "RNA"


@irdl_op_definition
class RoundTowardPositiveOp(RunningModeConstantOp):
    name = "smt.fp.round_toward_positive"

    def constant_name(self) -> str:
        return "roundTowardPositive"


@irdl_op_definition
class RTPOp(RunningModeConstantOp):
    name = "smt.fp.rtp"

    def constant_name(self) -> str:
        return "RTP"


@irdl_op_definition
class RoundTowardNegativeOp(RunningModeConstantOp):
    name = "smt.fp.round_toward_negative"

    def constant_name(self) -> str:
        return "roundTowardNegative"


@irdl_op_definition
class RTNOp(RunningModeConstantOp):
    name = "smt.fp.rtn"

    def constant_name(self) -> str:
        return "RTN"


@irdl_op_definition
class RoundTowardZeroOp(RunningModeConstantOp):
    name = "smt.fp.round_toward_zero"

    def constant_name(self) -> str:
        return "roundTowardZero"


@irdl_op_definition
class RTZOp(RunningModeConstantOp):
    name = "smt.fp.rtz"

    def constant_name(self) -> str:
        return "RTZ"


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
        return "NaN"


################################################################################
#                          FP Unary Ops                                        #
################################################################################
class UnaryFPOp(IRDLOperation, Pure, SimpleSMTLibOp):
    """
    A unary floating-point operation.
    """

    arg: Operand = operand_def(FloatingPointType)
    res: OpResult = result_def(FloatingPointType)

    def __init__(self, arg: SSAValue):
        super().__init__(result_types=[arg.type], operands=[arg])

    def verify_(self):
        if not (self.res.type == self.arg.type):
            raise VerifyException("Operand and result must have the same type")


@irdl_op_definition
class AbsOp(UnaryFPOp):
    name = "smt.fp.abs"

    def op_name(self) -> str:
        return "fp.abs"


@irdl_op_definition
class NegOp(UnaryFPOp):
    name = "smt.fp.neg"

    def op_name(self) -> str:
        return "fp.neg"


class UnaryFPOpWithRoundingMode(IRDLOperation, Pure, SimpleSMTLibOp):
    """
    A binary floating-point operation with a rounding mode operand.
    """

    roundingMode: Operand = operand_def(RoundingModeType)
    arg: Operand = operand_def(FloatingPointType)
    res: OpResult = result_def(FloatingPointType)

    def __init__(self, roundingMode: SSAValue, arg: SSAValue):
        super().__init__(result_types=[arg.type], operands=[roundingMode, arg])

    def verify_(self):
        if not (self.res.type == self.arg.type):
            raise VerifyException("Operand and result must have the same type")


@irdl_op_definition
class SqrtOp(UnaryFPOpWithRoundingMode):
    name = "smt.fp.sqrt"

    def op_name(self) -> str:
        return "fp.sqrt"


@irdl_op_definition
class RoundToIntegralOp(UnaryFPOpWithRoundingMode):
    name = "smt.fp.roundToIntegral"

    def op_name(self) -> str:
        return "fp.roundToIntegral"


################################################################################
#                          FP Binary Ops                                       #
################################################################################
class BinaryFPOp(IRDLOperation, Pure, SimpleSMTLibOp):
    """
    A binary floating-point operation.
    """

    lhs: Operand = operand_def(FloatingPointType)
    rhs: Operand = operand_def(FloatingPointType)
    res: OpResult = result_def(FloatingPointType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[lhs.type], operands=[lhs, rhs])

    def verify_(self):
        if not (self.res.type == self.lhs.type):
            raise VerifyException("Operand and result must have the same type")
        if not (self.lhs.type == self.rhs.type):
            raise VerifyException("LHS and RHS must have the same type")


@irdl_op_definition
class MaxOp(BinaryFPOp):
    name = "smt.fp.max"

    def op_name(self) -> str:
        return "fp.max"


@irdl_op_definition
class MinOp(BinaryFPOp):
    name = "smt.fp.min"

    def op_name(self) -> str:
        return "fp.min"


@irdl_op_definition
class RemOp(BinaryFPOp):
    name = "smt.fp.rem"

    def op_name(self) -> str:
        return "fp.rem"


class BinaryFPOpWithRoundingMode(IRDLOperation, Pure, SimpleSMTLibOp):
    """
    A binary floating-point operation with a rounding mode operand.
    """

    roundingMode: Operand = operand_def(RoundingModeType)
    lhs: Operand = operand_def(FloatingPointType)
    rhs: Operand = operand_def(FloatingPointType)
    res: OpResult = result_def(FloatingPointType)

    def __init__(self, roundingMode: SSAValue, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[lhs.type], operands=[roundingMode, lhs, rhs])

    def verify_(self):
        if not (self.res.type == self.lhs.type):
            raise VerifyException("Operand and result must have the same type")
        if not (self.lhs.type == self.rhs.type):
            raise VerifyException("LHS and RHS must have the same type")


@irdl_op_definition
class AddOp(BinaryFPOpWithRoundingMode):
    name = "smt.fp.add"

    def op_name(self) -> str:
        return "fp.add"


@irdl_op_definition
class SubOp(BinaryFPOpWithRoundingMode):
    name = "smt.fp.sub"

    def op_name(self) -> str:
        return "fp.sub"


@irdl_op_definition
class MulOp(BinaryFPOpWithRoundingMode):
    name = "smt.fp.mul"

    def op_name(self) -> str:
        return "fp.mul"


@irdl_op_definition
class DivOp(BinaryFPOpWithRoundingMode):
    name = "smt.fp.div"

    def op_name(self) -> str:
        return "fp.div"


################################################################################
#                          FP Unary Predicates                                 #
################################################################################
class UnaryFPPredicate(IRDLOperation, Pure, SimpleSMTLibOp):
    """
    A unary predicate for floating-points.
    """

    arg: Operand = operand_def(FloatingPointType)
    res: OpResult = result_def(BoolType)

    def __init__(self, arg: SSAValue):
        super().__init__(result_types=[BoolType()], operands=[arg])


@irdl_op_definition
class IsNormalOp(UnaryFPPredicate):
    name = "smt.fp.isNormal"

    def op_name(self) -> str:
        return "fp.isNormal"


@irdl_op_definition
class IsSubnormalOp(UnaryFPPredicate):
    name = "smt.fp.isSubnormal"

    def op_name(self) -> str:
        return "fp.isSubnormal"


@irdl_op_definition
class IsZeroOp(UnaryFPPredicate):
    name = "smt.fp.isZero"

    def op_name(self) -> str:
        return "fp.isZero"


@irdl_op_definition
class IsInfiniteOp(UnaryFPPredicate):
    name = "smt.fp.isInfinite"

    def op_name(self) -> str:
        return "fp.isInfinite"


@irdl_op_definition
class IsNaNOp(UnaryFPPredicate):
    name = "smt.fp.isNaN"

    def op_name(self) -> str:
        return "fp.isNaN"


@irdl_op_definition
class IsNegativeOp(UnaryFPPredicate):
    name = "smt.fp.isNegative"

    def op_name(self) -> str:
        return "fp.isNegative"


@irdl_op_definition
class IsPositiveOp(UnaryFPPredicate):
    name = "smt.fp.isPositive"

    def op_name(self) -> str:
        return "fp.isPositive"


################################################################################
#                          FP Binary Predicates                                #
################################################################################
class BinaryFPPredicate(IRDLOperation, Pure, SimpleSMTLibOp):
    """
    A binary predicate for floating-points.
    """

    lhs: Operand = operand_def(FloatingPointType)
    rhs: Operand = operand_def(FloatingPointType)
    res: OpResult = result_def(BoolType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[BoolType()], operands=[lhs, rhs])

    def verify_(self):
        if not (self.lhs.type == self.rhs.type):
            raise VerifyException("Operands must have the same type")


@irdl_op_definition
class EqOp(BinaryFPPredicate):
    name = "smt.fp.eq"

    def op_name(self) -> str:
        return "fp.eq"


@irdl_op_definition
class LeqOp(BinaryFPPredicate):
    name = "smt.fp.leq"

    def op_name(self) -> str:
        return "fp.leq"


@irdl_op_definition
class LtOp(BinaryFPPredicate):
    name = "smt.fp.lt"

    def op_name(self) -> str:
        return "fp.lt"


@irdl_op_definition
class GeqOp(BinaryFPPredicate):
    name = "smt.fp.geq"

    def op_name(self) -> str:
        return "fp.geq"


@irdl_op_definition
class GtOp(BinaryFPPredicate):
    name = "smt.fp.gt"

    def op_name(self) -> str:
        return "fp.gt"


SMTFloatingPointDialect = Dialect(
    "smt.fp",
    [
        ConstantOp,
        PositiveZeroOp,
        NegativeZeroOp,
        PositiveInfinityOp,
        NegativeInfinityOp,
        NaNOp,
        # Rounding Mode constants
        RoundNearestTiesToEvenOp,
        RNEOp,
        RoundNearestTiesToAwayOp,
        RNAOp,
        RoundTowardPositiveOp,
        RTPOp,
        RoundTowardNegativeOp,
        RTNOp,
        RoundTowardZeroOp,
        RTZOp,
        # Unary ops
        AbsOp,
        NegOp,
        SqrtOp,
        RoundToIntegralOp,
        # Binary ops,
        MaxOp,
        MinOp,
        RemOp,
        AddOp,
        SubOp,
        MulOp,
        DivOp,
        # Unary predicates
        IsNormalOp,
        IsSubnormalOp,
        IsZeroOp,
        IsInfiniteOp,
        IsNaNOp,
        IsNegativeOp,
        IsPositiveOp,
        # Binary predicates
        EqOp,
        LeqOp,
        LtOp,
        GeqOp,
        GtOp,
    ],
    [FloatingPointType, RoundingModeType],
)
