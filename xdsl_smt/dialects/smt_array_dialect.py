from __future__ import annotations

from typing import Annotated, Generic, TypeVar, IO

from xdsl.ir import (
    Attribute,
    Dialect,
    OpResult,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    operand_def,
    result_def,
    Operand,
    ParameterDef,
    irdl_op_definition,
    irdl_attr_definition,
    IRDLOperation,
    ConstraintVar,
)
from xdsl.utils.isattr import isattr

from xdsl_smt.traits.smt_printer import SMTLibSort, SimpleSMTLibOp
from xdsl_smt.traits.effects import Pure


DomainT = TypeVar("DomainT", bound=Attribute, covariant=True)
RangeT = TypeVar("RangeT", bound=Attribute, covariant=True)


@irdl_attr_definition
class ArrayType(
    Generic[DomainT, RangeT], ParametrizedAttribute, SMTLibSort, TypeAttribute
):
    name = "smt.array.array"
    domain: ParameterDef[DomainT]
    range: ParameterDef[RangeT]

    def print_sort_to_smtlib(self, stream: IO[str]):
        assert isinstance(self.domain, SMTLibSort)
        assert isinstance(self.range, SMTLibSort)
        print(f"(Array ", file=stream, end="")
        self.domain.print_sort_to_smtlib(stream)
        print(" ", file=stream, end="")
        self.range.print_sort_to_smtlib(stream)
        print(")", file=stream)

    def __init__(self, domain: Attribute, range: Attribute):
        super().__init__([domain, range])


AnyArrayType = ArrayType[Attribute, Attribute]


@irdl_op_definition
class SelectOp(IRDLOperation, SimpleSMTLibOp, Pure):
    name = "smt.array.select"

    _Domain = Annotated[Attribute, ConstraintVar("Domain")]
    _Range = Annotated[Attribute, ConstraintVar("Range")]

    array: Operand = operand_def(ArrayType[_Domain, _Range])
    index: Operand = operand_def(_Domain)

    res: OpResult = result_def(_Range)

    def __init__(
        self, array: SSAValue, index: SSAValue, result_type: Attribute | None = None
    ):
        if result_type is None:
            if not isattr(array.type, AnyArrayType):
                raise ValueError(
                    f"Expected array type, got {array.type} for array operand"
                )
            result_type = array.type.range
        super().__init__(result_types=[result_type], operands=[array, index])

    def op_name(self) -> str:
        return "select"

    assembly_format = "$array `[` $index `]` attr-dict `:` type($array)"


@irdl_op_definition
class StoreOp(IRDLOperation, SimpleSMTLibOp, Pure):
    name = "smt.array.store"

    _Domain = Annotated[Attribute, ConstraintVar("Domain")]
    _Range = Annotated[Attribute, ConstraintVar("Range")]
    _Array = Annotated[ArrayType[_Domain, _Range], ConstraintVar("Array")]

    array: Operand = operand_def(_Array)
    index: Operand = operand_def(_Domain)
    value: Operand = operand_def(_Range)

    res: OpResult = result_def(_Array)

    def __init__(self, array: SSAValue, index: SSAValue, value: SSAValue):
        if not isattr(array.type, AnyArrayType):
            raise ValueError(f"Expected array type, got {array.type} for array operand")
        super().__init__(result_types=[array.type], operands=[array, index, value])

    def op_name(self) -> str:
        return "store"

    assembly_format = "$array `[` $index `]` `,` $value attr-dict `:` type($array)"


SMTArray = Dialect("smt.array", [SelectOp, StoreOp], [ArrayType])
