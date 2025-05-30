from __future__ import annotations
from typing_extensions import TypeVar
from typing import Annotated, Generic, Sequence, ClassVar

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
    VarConstraint,
    AnyAttr,
    ParamAttrConstraint,
    GenericAttrConstraint,
    traits_def,
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
from xdsl.utils.hints import isa

_UBAttrParameter = TypeVar(
    "_UBAttrParameter", bound=Attribute, covariant=True, default=Attribute
)


@irdl_attr_definition
class UBOrType(Generic[_UBAttrParameter], ParametrizedAttribute, TypeAttribute):
    """A tagged union between a type and a UB singleton."""

    name = "ub.ub_or"

    type: ParameterDef[_UBAttrParameter]

    def __init__(self, type: _UBAttrParameter):
        super().__init__([type])

    @classmethod
    def constr(
        cls,
        type: GenericAttrConstraint[_UBAttrParameter],
    ) -> GenericAttrConstraint[UBOrType[_UBAttrParameter]]:
        return ParamAttrConstraint[UBOrType[_UBAttrParameter]](UBOrType, [type])


@irdl_op_definition
class UBOp(IRDLOperation):
    """Create an UB value."""

    name = "ub.ub"

    res = result_def(UBOrType)

    assembly_format = "attr-dict `:` type($res)"

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

    T: ClassVar = VarConstraint("T", AnyAttr())

    value = operand_def(T)
    res = result_def(UBOrType.constr(T))

    assembly_format = "$value attr-dict `:` type($res)"

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

    values = var_operand_def(UBOrType[T])

    value_region = region_def(single_block="single_block")
    ub_region = region_def(single_block="single_block")

    res = var_result_def()

    assembly_format = "$values attr-dict-with-keyword `:` `(` type($values) `)` `->` type($res) $value_region $ub_region"

    def __init__(self, values: Sequence[SSAValue], result_types: Sequence[Attribute]):
        value_types = list[UBOrType[Attribute]]()
        for value in values:
            if not isa(value.type, UBOrType[Attribute]):
                raise ValueError(f"Expected a '{UBOrType.name}' type, got {value.type}")
            value_types.append(value.type)
        value_region = Region(Block((), arg_types=value_types))
        ub_region = Region(Block((), arg_types=[]))
        super().__init__(
            operands=[values],
            result_types=[result_types],
            regions=[value_region, ub_region],
        )

    @property
    def value_terminator(self) -> YieldOp:
        if not isinstance(self.value_region.block.last_op, YieldOp):
            raise ValueError("Value case region must have a yield terminator")
        return self.value_region.block.last_op

    @property
    def ub_terminator(self) -> YieldOp:
        if not isinstance(self.ub_region.block.last_op, YieldOp):
            raise ValueError("UB case region must have a yield terminator")
        return self.ub_region.block.last_op

    @property
    def value_types(self) -> Sequence[UBOrType[Attribute]]:
        types = list[UBOrType[Attribute]]()
        for value in self.values:
            assert isa(value.type, UBOrType[Attribute])
            types.append(value.type)
        return types

    def verify_(self):
        value_type_type = [type.type for type in self.value_types]
        if list(self.value_region.blocks[0].arg_types) != value_type_type:
            raise ValueError(
                "Value region must have exactly one argument of type "
                f"{tuple(value_type_type)}, got {tuple(self.value_region.blocks[0].arg_types)}"
            )
        if len(self.ub_region.blocks[0].args) != 0:
            raise ValueError("UB region must have no arguments")
        if not isinstance(term_val := self.value_region.blocks[0].last_op, YieldOp):
            raise ValueError(
                f"Value region must end with a 'ub.yield' op, got {term_val}"
            )
        if not isinstance(term_ub := self.ub_region.blocks[0].last_op, YieldOp):
            raise ValueError(f"UB region must end with a 'ub.yield' op, got {term_ub}")
        if self.result_types != term_val.operand_types:
            raise ValueError(
                "Yielded types in value region must match the result types, "
                f"got {term_val.operand_types} and {self.result_types}"
            )
        if self.result_types != term_ub.operand_types:
            raise ValueError(
                "Yielded types in UB region must match the result types, "
                f"got {term_ub.operand_types} and {self.result_types}"
            )


@irdl_op_definition
class YieldOp(IRDLOperation):
    """Yield a value inside an `ub.match` region."""

    name = "ub.yield"

    rets = var_operand_def()

    assembly_format = "$rets attr-dict `:` type($rets)"

    traits = traits_def(IsTerminator())

    def __init__(self, *rets: SSAValue):
        super().__init__(
            operands=list(rets),
            result_types=[],
        )


UBDialect = Dialect("ub", [UBOp, FromOp, MatchOp, YieldOp], [UBOrType])
