from __future__ import annotations

from functools import reduce
from typing import Generic, Sequence, TypeAlias, TypeVar, cast, IO
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
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

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

    def __init__(self, first: SSAValue, second: SSAValue) -> None:
        result_type = PairType(first.type, second.type)
        return super().__init__(result_types=[result_type], operands=[first, second])

    def verify_(self):
        assert isinstance(self.res.type, PairType)
        res_typ = cast(AnyPairType, self.res.type)
        if res_typ.first != self.first.type or res_typ.second != self.second.type:
            raise VerifyException(
                f"{self.name} result type is incompatible with operand types."
            )

    def op_name(self) -> str:
        return "pair"


@irdl_op_definition
class FirstOp(IRDLOperation, Pure, SimpleSMTLibOp):
    name = "smt.utils.first"

    res: OpResult = result_def()
    pair: Operand = operand_def(AnyPairType)

    def __init__(self, pair: SSAValue) -> None:
        if not isinstance(pair.type, PairType):
            raise VerifyException(
                f"{self.name} operand is expected to be a {PairType.name} type"
            )
        pair_typ = cast(AnyPairType, pair.type)
        super().__init__(result_types=[pair_typ.first], operands=[pair])

    def verify_(self):
        assert isinstance(self.pair.type, PairType)
        pair_typ = cast(AnyPairType, self.pair.type)
        if self.res.type != pair_typ.first:
            raise VerifyException(
                f"{self.name} result type is incompatible with operand types."
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
                f"{self.name} result type is incompatible with operand types."
            )

    def __init__(self, pair: SSAValue) -> None:
        if not isinstance(pair.type, PairType):
            raise VerifyException(
                f"{self.name} operand is expected to be a {PairType.name} type"
            )
        pair_typ = cast(AnyPairType, pair.type)
        return super().__init__(result_types=[pair_typ.second], operands=[pair])


def pair_from_list(*vals: SSAValue) -> SSAValue:
    """Convert a list of values into a cons-list of SMT pairs"""

    if len(vals) == 0:
        raise ValueError("Must have at least one value")
    elif len(vals) == 1:
        return vals[0]
    else:
        return reduce(lambda r, l: SSAValue.get(PairOp(l, r)), reversed(vals))


def merge_values_with_pairs(
    vals: Sequence[SSAValue], rewriter: PatternRewriter, insert_point: InsertPoint
) -> SSAValue | None:
    """Merge a list of values (a, b, c, ...) into a list of pairs (a, (b, (c, ...)))"""
    if len(vals) == 0:
        return None
    rhs = vals[-1]
    for lhs in reversed(vals[:-1]):
        pair = PairOp(lhs, rhs)
        rewriter.insert_op(pair, insert_point)
        rhs = pair.res
    return rhs


def pair_type_from_list(*types: Attribute) -> Attribute:
    """Convert a list of types into a cons-list of SMT pair types."""

    if len(types) == 0:
        raise ValueError("Must have at least one value")
    if len(types) == 1:
        return types[0]
    return reduce(lambda r, l: AnyPairType(l, r), reversed(types))


SMTUtilsDialect = Dialect("smt.utils", [PairOp, FirstOp, SecondOp], [PairType])
