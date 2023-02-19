from __future__ import annotations

from io import IOBase
from typing import Annotated, Generic, TypeAlias, TypeVar, cast
from xdsl.ir import (Attribute, Dialect, OpResult, Operation,
                     ParametrizedAttribute, SSAValue)
from xdsl.utils.exceptions import VerifyException
from xdsl.irdl import (Operand, ParameterDef, irdl_attr_definition,
                       irdl_op_definition)

from .smt_dialect import SMTLibSort, SimpleSMTLibOp
from traits.effects import Pure

_F = TypeVar("_F", bound=Attribute)
_S = TypeVar("_S", bound=Attribute)


@irdl_attr_definition
class PairType(Generic[_F, _S], ParametrizedAttribute, SMTLibSort):
    name = "smt.utils.pair"

    first: ParameterDef[_F]
    second: ParameterDef[_S]

    def print_sort_to_smtlib(self, stream: IOBase) -> None:
        assert isinstance(self.first, SMTLibSort)
        assert isinstance(self.second, SMTLibSort)
        print("(Pair ", file=stream, end='')
        self.first.print_sort_to_smtlib(stream)
        print(" ", file=stream, end='')
        self.second.print_sort_to_smtlib(stream)
        print(")", file=stream, end='')

    @staticmethod
    def from_params(first: _F, second: _S) -> PairType[_F, _S]:
        return PairType([first, second])


AnyPairType: TypeAlias = PairType[Attribute, Attribute]


@irdl_op_definition
class PairOp(Operation, Pure, SimpleSMTLibOp):
    name = "smt.utils.pair"

    res: Annotated[OpResult, AnyPairType]
    first: Operand
    second: Operand

    @staticmethod
    def from_values(first: SSAValue, second: SSAValue) -> PairOp:
        result_type = PairType.from_params(first.typ, second.typ)
        return PairOp.create(result_types=[result_type],
                             operands=[first, second])

    def verify_(self):
        assert isinstance(self.res.typ, PairType)
        res_typ = cast(AnyPairType, self.res.typ)
        if (res_typ.first != self.first.typ
                or res_typ.second != self.second.typ):
            raise VerifyException(
                "{self.name} result type is incompatible with operand types.")

    def op_name(self) -> str:
        return "pair"


@irdl_op_definition
class FirstOp(Operation, Pure, SimpleSMTLibOp):
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

    def op_name(self) -> str:
        return "first"


@irdl_op_definition
class SecondOp(Operation, Pure, SimpleSMTLibOp):
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


SMTUtilsDialect = Dialect([PairOp, FirstOp, SecondOp], [PairType])
