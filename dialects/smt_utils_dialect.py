from __future__ import annotations

from typing import Annotated, Generic, TypeAlias, TypeVar, cast, IO
from xdsl.ir import (Attribute, Dialect, OpResult, ParametrizedAttribute,
                     SSAValue)
from xdsl.parser import BaseParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.irdl import (Operand, ParameterDef, irdl_attr_definition,
                       irdl_op_definition, IRDLOperation)

from .smt_dialect import SMTLibSort, SimpleSMTLibOp
from traits.effects import Pure

_F = TypeVar("_F", bound=Attribute, covariant=True)
_S = TypeVar("_S", bound=Attribute, covariant=True)


@irdl_attr_definition
class PairType(Generic[_F, _S], ParametrizedAttribute, SMTLibSort):
    name = "smt.utils.pair"

    first: ParameterDef[_F]
    second: ParameterDef[_S]

    def __init__(self: PairType[_F, _S], first: _F, second: _S):
        super().__init__([first, second])

    def print_sort_to_smtlib(self, stream: IO[str]) -> None:
        assert isinstance(self.first, SMTLibSort)
        assert isinstance(self.second, SMTLibSort)
        print("(Pair ", file=stream, end='')
        self.first.print_sort_to_smtlib(stream)
        print(" ", file=stream, end='')
        self.second.print_sort_to_smtlib(stream)
        print(")", file=stream, end='')


AnyPairType: TypeAlias = PairType[Attribute, Attribute]


@irdl_op_definition
class PairOp(IRDLOperation, Pure, SimpleSMTLibOp):
    name = "smt.utils.pair"

    res: Annotated[OpResult, AnyPairType]
    first: Operand
    second: Operand

    @staticmethod
    def from_values(first: SSAValue, second: SSAValue) -> PairOp:
        result_type = PairType(first.typ, second.typ)
        return PairOp.create(result_types=[result_type],
                             operands=[first, second])

    def verify_(self):
        assert isinstance(self.res.typ, PairType)
        res_typ = cast(AnyPairType, self.res.typ)
        if (res_typ.first != self.first.typ
                or res_typ.second != self.second.typ):
            raise VerifyException(
                "{self.name} result type is incompatible with operand types.")

    @classmethod
    def parse(cls, result_types: list[Attribute],
              parser: BaseParser) -> PairOp:
        first = parser.parse_operand()
        parser.parse_characters(",", "Expected `,`")
        second = parser.parse_operand()
        return PairOp.create(result_types=[PairType(first.typ, second.typ)],
                             operands=[first, second])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.first)
        printer.print(", ")
        printer.print_ssa_value(self.second)

    def op_name(self) -> str:
        return "pair"


@irdl_op_definition
class FirstOp(IRDLOperation, Pure, SimpleSMTLibOp):
    name = "smt.utils.first"

    res: OpResult
    pair: Annotated[Operand, AnyPairType]

    @staticmethod
    def from_value(pair: SSAValue) -> FirstOp:
        if not isinstance(pair.typ, PairType):
            raise VerifyException(
                "{self.name} operand is expected to be a {PairType.name} type")
        pair_typ = cast(AnyPairType, pair.typ)
        return FirstOp.create(result_types=[pair_typ.first], operands=[pair])

    def verify_(self):
        assert isinstance(self.pair.typ, PairType)
        pair_typ = cast(AnyPairType, self.pair.typ)
        if self.res.typ != pair_typ.first:
            raise VerifyException(
                "{self.name} result type is incompatible with operand types.")

    @classmethod
    def parse(cls, result_types: list[Attribute],
              parser: BaseParser) -> FirstOp:
        val = parser.parse_operand()
        assert (isinstance(val.typ, PairType))
        typ = cast(AnyPairType, val.typ)
        return cls.build(result_types=[typ.first], operands=[val])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.pair)

    def op_name(self) -> str:
        return "first"


@irdl_op_definition
class SecondOp(IRDLOperation, Pure, SimpleSMTLibOp):
    name = "smt.utils.second"

    res: OpResult
    pair: Annotated[Operand, AnyPairType]

    def op_name(self) -> str:
        return "second"

    def verify_(self):
        assert isinstance(self.pair.typ, PairType)
        pair_typ = cast(PairType[Attribute, Attribute], self.pair.typ)
        if self.res.typ != pair_typ.second:
            raise VerifyException(
                "{self.name} result type is incompatible with operand types.")

    @staticmethod
    def from_value(pair: SSAValue) -> SecondOp:
        if not isinstance(pair.typ, PairType):
            raise VerifyException(
                "{self.name} operand is expected to be a {PairType.name} type")
        pair_typ = cast(AnyPairType, pair.typ)
        return SecondOp.create(result_types=[pair_typ.second], operands=[pair])

    @classmethod
    def parse(cls, result_types: list[Attribute],
              parser: BaseParser) -> SecondOp:
        val = parser.parse_operand()
        assert (isinstance(val.typ, PairType))
        typ = cast(AnyPairType, val.typ)
        return cls.build(result_types=[typ.second], operands=[val])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.pair)


SMTUtilsDialect = Dialect([PairOp, FirstOp, SecondOp], [PairType])
