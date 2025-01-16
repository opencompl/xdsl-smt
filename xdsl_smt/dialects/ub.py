from __future__ import annotations
from typing import Annotated, Generic, TypeVar

from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    IRDLOperation,
    ParameterDef,
    ConstraintVar,
    region_def,
    var_result_def,
    var_operand_def,
)
from xdsl.ir import (
    ParametrizedAttribute,
    Attribute,
    SSAValue,
    Region,
    Block,
    Dialect,
    TypeAttribute,
)
from xdsl.traits import IsTerminator
from xdsl.utils.isattr import isattr

_UBAttrParameter = TypeVar("_UBAttrParameter", bound=Attribute)


@irdl_attr_definition
class UBOrType(Generic[_UBAttrParameter], ParametrizedAttribute, TypeAttribute):
    """A tagged union between a type and a UB singleton."""

    name = "ub.ub_or"

    type: ParameterDef[_UBAttrParameter]

    def __init__(self, type: _UBAttrParameter):
        super().__init__([type])


@irdl_op_definition
class UBOp(IRDLOperation):
    """Create an UB value."""

    name = "ub.ub"

    new_ub = result_def(UBOrType)

    assembly_format = "attr-dict `:` type($new_ub)"

    def __init__(self, type: Attribute):
        """Create an UB value for the given type."""
        super().__init__(
            operands=[],
            result_types=[UBOrType(type)],
        )


@irdl_op_definition
class FromOp(IRDLOperation):
    """Convert a value to a value + UB type."""

    name = "ub.from"

    T = Annotated[Attribute, ConstraintVar("T")]

    value = operand_def(T)
    result = result_def(UBOrType[T])

    assembly_format = "$value attr-dict `:` type($result)"

    def __init__(self, value: SSAValue):
        super().__init__(
            operands=[value],
            result_types=[UBOrType(value.type)],
        )


@irdl_op_definition
class MatchOp(IRDLOperation):
    """Pattern match on a tagged union between a value and UB."""

    name = "ub.match"

    T = Annotated[Attribute, ConstraintVar("T")]

    value = operand_def(UBOrType[T])

    value_region = region_def(single_block="single_block")
    ub_region = region_def(single_block="single_block")

    res = var_result_def()

    assembly_format = "$value attr-dict-with-keyword `:` type($value) `->` type($res) $value_region $ub_region"

    def __init__(self, value: SSAValue):
        if not isattr(value.type, UBOrType[Attribute]):
            raise ValueError(f"Expected a '{UBOrType.name}' type, got {value.type}")
        value_region = Region(Block((), arg_types=[value.type.type]))
        ub_region = Region(Block((), arg_types=[]))
        super().__init__(
            operands=[value],
            result_types=[],
            regions=[value_region, ub_region],
        )


@irdl_op_definition
class YieldOp(IRDLOperation):
    """Yield a value inside an `ub.match` region."""

    name = "ub.yield"

    rets = var_operand_def()

    assembly_format = "$rets attr-dict `:` type($rets)"

    traits = frozenset([IsTerminator()])

    def __init__(self, *rets: SSAValue):
        super().__init__(
            operands=list(rets),
            result_types=[],
        )


UBDialect = Dialect("ub", [UBOp, FromOp, MatchOp, YieldOp], [UBOrType])
