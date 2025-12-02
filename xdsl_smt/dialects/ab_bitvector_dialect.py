from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import IntAttr, IntegerAttr, IntegerType

from xdsl.ir import (
    Dialect,
    SSAValue,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    operand_def,
    result_def,
    irdl_op_definition,
    irdl_attr_definition,
    IRDLOperation,
    traits_def,
    prop_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import Pure

from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl.dialects.smt import BitVectorType


@irdl_attr_definition
class BitWidthType(ParametrizedAttribute, TypeAttribute):
    """The bitwidth of an arbitrary-bitvector type."""

    name = "abbv.bitwidth"


@irdl_attr_definition
class ArbitraryBitVectorType(ParametrizedAttribute, TypeAttribute):
    """An arbitrary-width bitvector type."""

    name = "abbv.bv"


@irdl_op_definition
class ConstantBitWidthOp(IRDLOperation):
    name = "abbv.constant_bitwidth"

    value = prop_def(IntAttr)
    result = result_def(BitWidthType())

    traits = traits_def(Pure())

    @classmethod
    def parse(cls, parser: Parser) -> ConstantBitWidthOp:
        value = parser.parse_integer()
        attr_dict = parser.parse_optional_attr_dict()
        op = ConstantBitWidthOp(value)
        if attr_dict:
            op.attributes = attr_dict
        return op

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_int(self.value.data)
        printer.print_string(" ")
        printer.print_attr_dict(self.properties)

    def __init__(
        self,
        value: int | IntAttr,
    ) -> None:
        if isinstance(value, int):
            value = IntAttr(value)
        super().__init__(result_types=[BitWidthType()], properties={"value": value})


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "abbv.constant"

    width = operand_def(BitWidthType)
    value = prop_def(IntAttr)
    result = result_def(ArbitraryBitVectorType())

    traits = traits_def(Pure())

    @classmethod
    def parse(cls, parser: Parser) -> ConstantOp:
        value = parser.parse_integer()
        parser.parse_characters(":")
        width = parser.parse_operand()
        return ConstantOp(width, value)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_int(self.value.data)
        printer.print_string(" : ")
        printer.print_operand(self.width)

    def __init__(
        self,
        width: SSAValue,
        value: int | IntAttr,
    ) -> None:
        if isinstance(value, int):
            value = IntAttr(value)
        super().__init__(
            result_types=[ArbitraryBitVectorType()],
            operands=[width],
            properties={"value": value},
        )


@irdl_op_definition
class FromFixedBitWidthOp(IRDLOperation):
    name = "abbv.from_fixed_bitwidth"

    operand = operand_def(BitVectorType)
    result = result_def(ArbitraryBitVectorType())

    traits = traits_def(Pure())

    assembly_format = "$operand attr-dict `:` type($operand)"

    def __init__(self, operand: SSAValue) -> None:
        super().__init__(result_types=[ArbitraryBitVectorType()], operands=[operand])


@irdl_op_definition
class ToFixedBitWidthOp(IRDLOperation):
    """
    Converts an arbitrary-width bitvector to a fixed-width bitvector.
    It is expected that the bitwidth of the arbitrary-width bitvector is equal to
    the bitwidth of the fixed-width bitvector.
    """

    name = "abbv.to_fixed_bitwidth"

    operand = operand_def(ArbitraryBitVectorType)
    result = result_def(BitVectorType)

    traits = traits_def(Pure())

    assembly_format = "$operand attr-dict `:` type($result)"

    def __init__(self, operand: SSAValue, result_type: BitVectorType) -> None:
        super().__init__(result_types=[result_type], operands=[operand])


@irdl_op_definition
class GetBitWidthOp(IRDLOperation):
    """
    Get the bitwidth of an arbitrary-width bitvector.
    """

    name = "abbv.get_bitwidth"

    operand = operand_def(ArbitraryBitVectorType)
    result = result_def(BitWidthType)

    traits = traits_def(Pure())

    assembly_format = "$operand attr-dict"

    def __init__(self, operand: SSAValue) -> None:
        super().__init__(result_types=[BitWidthType()], operands=[operand])


class UnaryBVOp(IRDLOperation):
    res = result_def(ArbitraryBitVectorType())
    arg = operand_def(ArbitraryBitVectorType())

    traits = traits_def(Pure())

    assembly_format = "$arg attr-dict"

    def __init__(self, arg: SSAValue):
        super().__init__(result_types=[arg.type], operands=[arg])


class BinaryBVOp(IRDLOperation):
    res = result_def(ArbitraryBitVectorType())
    lhs = operand_def(ArbitraryBitVectorType())
    rhs = operand_def(ArbitraryBitVectorType())

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs attr-dict"

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[ArbitraryBitVectorType()], operands=[lhs, rhs])


################################################################################
#                          Basic Bitvector Arithmetic                          #
################################################################################


@irdl_op_definition
class AddOp(BinaryBVOp):
    name = "abbv.add"

    traits = traits_def(Pure())


@irdl_op_definition
class SubOp(BinaryBVOp):
    name = "abbv.sub"

    traits = traits_def(Pure())


@irdl_op_definition
class NegOp(UnaryBVOp):
    name = "abbv.neg"

    traits = traits_def(Pure())


@irdl_op_definition
class MulOp(BinaryBVOp):
    name = "abbv.mul"

    traits = traits_def(Pure())


@irdl_op_definition
class URemOp(BinaryBVOp):
    name = "abbv.urem"

    traits = traits_def(Pure())


@irdl_op_definition
class SRemOp(BinaryBVOp):
    name = "abbv.srem"

    traits = traits_def(Pure())


@irdl_op_definition
class SModOp(BinaryBVOp):
    name = "abbv.smod"

    traits = traits_def(Pure())


@irdl_op_definition
class ShlOp(BinaryBVOp):
    name = "abbv.shl"

    traits = traits_def(Pure())


@irdl_op_definition
class LShrOp(BinaryBVOp):
    name = "abbv.lshr"

    traits = traits_def(Pure())


@irdl_op_definition
class AShrOp(BinaryBVOp):
    name = "abbv.ashr"

    traits = traits_def(Pure())


@irdl_op_definition
class SDivOp(BinaryBVOp):
    name = "abbv.sdiv"

    traits = traits_def(Pure())


@irdl_op_definition
class UDivOp(BinaryBVOp):
    name = "abbv.udiv"

    traits = traits_def(Pure())


################################################################################
#                                   Bitwise                                    #
################################################################################


@irdl_op_definition
class OrOp(BinaryBVOp):
    name = "abbv.or"

    traits = traits_def(Pure())


@irdl_op_definition
class AndOp(BinaryBVOp):
    name = "abbv.and"

    traits = traits_def(Pure())


@irdl_op_definition
class NotOp(UnaryBVOp):
    name = "abbv.not"

    traits = traits_def(Pure())


@irdl_op_definition
class XorOp(BinaryBVOp):
    name = "abbv.xor"

    traits = traits_def(Pure())


@irdl_op_definition
class NAndOp(BinaryBVOp):
    name = "abbv.nand"

    traits = traits_def(Pure())


@irdl_op_definition
class NorOp(BinaryBVOp):
    name = "abbv.nor"

    traits = traits_def(Pure())


@irdl_op_definition
class XNorOp(BinaryBVOp):
    name = "abbv.xnor"

    traits = traits_def(Pure())


################################################################################
#                                  Predicate                                   #
################################################################################


class UnaryPredBVOp(IRDLOperation):
    res = result_def(BoolType())
    operand = operand_def(ArbitraryBitVectorType())

    traits = traits_def(Pure())

    assembly_format = "$operand attr-dict"

    def __init__(self, operand: SSAValue):
        super().__init__(result_types=[BoolType()], operands=[operand])


class BinaryPredBVOp(IRDLOperation):
    res = result_def(BoolType())
    lhs = operand_def(ArbitraryBitVectorType())
    rhs = operand_def(ArbitraryBitVectorType())

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs attr-dict"

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[BoolType()], operands=[lhs, rhs])


@irdl_op_definition
class CmpOp(IRDLOperation):
    name = "abbv.cmp"

    res = result_def(BoolType())
    lhs = operand_def(ArbitraryBitVectorType())
    rhs = operand_def(ArbitraryBitVectorType())

    pred = prop_def(IntegerAttr[IntegerType])

    traits = traits_def(Pure())

    PREDICATE_NAME: ClassVar[dict[int, str]] = {
        0: "slt",
        1: "sle",
        2: "sgt",
        3: "sge",
        4: "ult",
        5: "ule",
        6: "ugt",
        7: "uge",
    }

    def __init__(self, lhs: SSAValue, rhs: SSAValue, pred: int | str | IntegerAttr):
        if isinstance(pred, str):
            pred = IntegerAttr(
                list(self.PREDICATE_NAME.keys())[
                    list(self.PREDICATE_NAME.values()).index(pred)
                ],
                IntegerType(64),
            )
        if isinstance(pred, int):
            pred = IntegerAttr(pred, IntegerType(64))
        super().__init__(
            result_types=[BoolType()], operands=[lhs, rhs], properties={"pred": pred}
        )


@irdl_op_definition
class UmulNoOverflowOp(BinaryPredBVOp):
    name = "abbv.umul_noovfl"

    traits = traits_def(Pure())


@irdl_op_definition
class SmulNoOverflowOp(BinaryPredBVOp):
    name = "abbv.smul_noovfl"

    traits = traits_def(Pure())


@irdl_op_definition
class SmulNoUnderflowOp(BinaryPredBVOp):
    name = "abbv.smul_noudfl"

    traits = traits_def(Pure())


@irdl_op_definition
class NegOverflowOp(UnaryPredBVOp):
    name = "abbv.nego"

    traits = traits_def(Pure())


@irdl_op_definition
class UaddOverflowOp(BinaryPredBVOp):
    name = "abbv.uaddo"

    traits = traits_def(Pure())


@irdl_op_definition
class SaddOverflowOp(BinaryPredBVOp):
    name = "abbv.saddo"

    traits = traits_def(Pure())


@irdl_op_definition
class UmulOverflowOp(BinaryPredBVOp):
    name = "abbv.umulo"

    traits = traits_def(Pure())


@irdl_op_definition
class SmulOverflowOp(BinaryPredBVOp):
    name = "abbv.smulo"

    traits = traits_def(Pure())


################################################################################
#                                    Others                                    #
################################################################################


@irdl_op_definition
class ConcatOp(IRDLOperation):
    name = "abbv.concat"

    lhs = operand_def(ArbitraryBitVectorType())
    rhs = operand_def(ArbitraryBitVectorType())
    res = result_def(ArbitraryBitVectorType())

    assembly_format = "$lhs `,` $rhs attr-dict"

    traits = traits_def(Pure())

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[ArbitraryBitVectorType()], operands=[lhs, rhs])


@irdl_op_definition
class ZeroExtendOp(IRDLOperation):
    name = "abbv.zero_extend"

    operand = operand_def(ArbitraryBitVectorType())
    width = operand_def(BitWidthType)
    res = result_def(ArbitraryBitVectorType())

    assembly_format = "$operand attr-dict `:` $width"

    traits = traits_def(Pure())

    def __init__(self, operand: SSAValue, width: SSAValue):
        super().__init__(
            result_types=[ArbitraryBitVectorType()], operands=[operand, width]
        )


@irdl_op_definition
class SignExtendOp(IRDLOperation):
    name = "abbv.sign_extend"

    operand = operand_def(ArbitraryBitVectorType())
    width = operand_def(BitWidthType)
    res = result_def(ArbitraryBitVectorType())

    assembly_format = "$operand attr-dict `:` $width"

    traits = traits_def(Pure())

    def __init__(self, operand: SSAValue, width: SSAValue):
        super().__init__(
            result_types=[ArbitraryBitVectorType()], operands=[operand, width]
        )


ABBitVectorDialect = Dialect(
    "abbv",
    [
        ConstantBitWidthOp,
        ConstantOp,
        FromFixedBitWidthOp,
        ToFixedBitWidthOp,
        GetBitWidthOp,
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
        XorOp,
        AndOp,
        NAndOp,
        NorOp,
        XNorOp,
        # Predicate
        CmpOp,
        # Overflow Predicate
        NegOverflowOp,
        UaddOverflowOp,
        SaddOverflowOp,
        UmulOverflowOp,
        SmulOverflowOp,
        UmulNoOverflowOp,
        SmulNoOverflowOp,
        SmulNoUnderflowOp,
        # Others
        ConcatOp,
        ZeroExtendOp,
        SignExtendOp,
    ],
    [ArbitraryBitVectorType, BitWidthType],
)
