from __future__ import annotations

from functools import reduce
from typing import Generic, TypeAlias, TypeVar, cast, IO
from xdsl.ir import (
    Attribute,
    Dialect,
    OpResult,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.irdl import (
    operand_def,
    result_def,
    Operand,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    IRDLOperation,
)

from .smt_dialect import SMTLibSort, SimpleSMTLibOp
from ..traits.effects import Pure

_F = TypeVar("_F", bound=Attribute, covariant=True)
_S = TypeVar("_S", bound=Attribute, covariant=True)


@irdl_attr_definition
class PairType(Generic[_F, _S], ParametrizedAttribute, SMTLibSort, TypeAttribute):
    name = "smt.utils.pair"

    first: ParameterDef[_F]
    second: ParameterDef[_S]

    def __init__(self: PairType[_F, _S], first: _F, second: _S):
        super().__init__([first, second])

    def print_sort_to_smtlib(self, stream: IO[str]) -> None:
        assert isinstance(self.first, SMTLibSort)
        assert isinstance(self.second, SMTLibSort)
        print("(Pair ", file=stream, end="")
        self.first.print_sort_to_smtlib(stream)
        print(" ", file=stream, end="")
        self.second.print_sort_to_smtlib(stream)
        print(")", file=stream, end="")


AnyPairType: TypeAlias = PairType[Attribute, Attribute]


@irdl_op_definition
class PairOp(IRDLOperation, Pure, SimpleSMTLibOp):
    name = "smt.utils.pair"

    res: OpResult = result_def(AnyPairType)
    first: Operand = operand_def()
    second: Operand = operand_def()

    @staticmethod
    def from_values(first: SSAValue, second: SSAValue) -> PairOp:
        result_type = PairType(first.type, second.type)
        return PairOp.create(result_types=[result_type], operands=[first, second])

    def verify_(self):
        assert isinstance(self.res.type, PairType)
        res_typ = cast(AnyPairType, self.res.type)
        if res_typ.first != self.first.type or res_typ.second != self.second.type:
            raise VerifyException(
                "{self.name} result type is incompatible with operand types."
            )

    def op_name(self) -> str:
        return "pair"


@irdl_op_definition
class FirstOp(IRDLOperation, Pure, SimpleSMTLibOp):
    name = "smt.utils.first"

    res: OpResult = result_def()
    pair: Operand = operand_def(AnyPairType)

    @staticmethod
    def from_value(pair: SSAValue) -> FirstOp:
        if not isinstance(pair.type, PairType):
            raise VerifyException(
                "{self.name} operand is expected to be a {PairType.name} type"
            )
        pair_typ = cast(AnyPairType, pair.type)
        return FirstOp.create(result_types=[pair_typ.first], operands=[pair])

    def verify_(self):
        assert isinstance(self.pair.type, PairType)
        pair_typ = cast(AnyPairType, self.pair.type)
        if self.res.type != pair_typ.first:
            raise VerifyException(
                "{self.name} result type is incompatible with operand types."
            )

    def op_name(self) -> str:
        return "first"


@irdl_op_definition
class SecondOp(IRDLOperation, Pure, SimpleSMTLibOp):
    name = "smt.utils.second"

    res: OpResult = result_def()
    pair: Operand = operand_def(AnyPairType)

    def op_name(self) -> str:
        return "second"

    def verify_(self):
        assert isinstance(self.pair.type, PairType)
        pair_typ = cast(PairType[Attribute, Attribute], self.pair.type)
        if self.res.type != pair_typ.second:
            raise VerifyException(
                "{self.name} result type is incompatible with operand types."
            )

    @staticmethod
    def from_value(pair: SSAValue) -> SecondOp:
        if not isinstance(pair.type, PairType):
            raise VerifyException(
                "{self.name} operand is expected to be a {PairType.name} type"
            )
        pair_typ = cast(AnyPairType, pair.type)
        return SecondOp.create(result_types=[pair_typ.second], operands=[pair])


def pair_from_list(*vals: SSAValue) -> SSAValue:
    """Convert a list of values into a cons-list of SMT pairs"""

    if len(vals) == 0:
        raise ValueError("Must have at least one value")
    elif len(vals) == 1:
        return vals[0]
    else:
        return reduce(
            lambda r, l: SSAValue.get(PairOp.from_values(l, r)), reversed(vals)
        )


SMTUtilsDialect = Dialect([PairOp, FirstOp, SecondOp], [PairType])
