from __future__ import annotations

from typing_extensions import TypeVar
from typing import ClassVar, Generic, IO

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
    traits_def,
    VarConstraint,
    AnyAttr,
    GenericAttrConstraint,
    ParamAttrConstraint,
)
from xdsl.parser import Parser
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import Pure, HasCanonicalizationPatternsTrait
from xdsl.utils.hints import isa

from xdsl_smt.traits.smt_printer import SMTConversionCtx, SMTLibSort, SimpleSMTLibOp


DomainT = TypeVar("DomainT", bound=Attribute, covariant=True, default=Attribute)
RangeT = TypeVar("RangeT", bound=Attribute, covariant=True, default=Attribute)


@irdl_attr_definition
class ArrayType(
    Generic[DomainT, RangeT], ParametrizedAttribute, SMTLibSort, TypeAttribute
):
    name = "smt.array.array"
    domain: ParameterDef[DomainT]
    range: ParameterDef[RangeT]

    @classmethod
    def constr(
        cls,
        domain: GenericAttrConstraint[DomainT],
        range: GenericAttrConstraint[RangeT],
    ) -> GenericAttrConstraint[ArrayType[DomainT, RangeT]]:
        return ParamAttrConstraint[ArrayType[DomainT, RangeT]](
            ArrayType, [domain, range]
        )

    def print_sort_to_smtlib(self, stream: IO[str]):
        print(f"(Array ", file=stream, end="")
        SMTConversionCtx.print_sort_to_smtlib(self.domain, stream)
        print(" ", file=stream, end="")
        SMTConversionCtx.print_sort_to_smtlib(self.range, stream)
        print(")", file=stream)

    def __init__(self, domain: DomainT, range: RangeT):
        super().__init__([domain, range])


AnyArrayType = ArrayType[Attribute, Attribute]


class SelectCanonPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_array import (
            SelectStorePattern,
        )

        return (SelectStorePattern(),)


@irdl_op_definition
class SelectOp(IRDLOperation, SimpleSMTLibOp):
    name = "smt.array.select"

    traits = traits_def(Pure(), SelectCanonPatterns())

    _D: ClassVar = VarConstraint("_D", AnyAttr())
    _R: ClassVar = VarConstraint("_R", AnyAttr())

    array: Operand = operand_def(ArrayType[Attribute, Attribute].constr(_D, _R))
    index: Operand = operand_def(_D)

    res: OpResult = result_def(_R)

    def __init__(
        self, array: SSAValue, index: SSAValue, result_type: Attribute | None = None
    ):
        if result_type is None:
            if not isa(array.type, AnyArrayType):
                raise ValueError(
                    f"Expected array type, got {array.type} for array operand"
                )
            result_type = array.type.range
        super().__init__(result_types=[result_type], operands=[array, index])

    def op_name(self) -> str:
        return "select"

    assembly_format = "$array `[` $index `]` attr-dict `:` type($array)"


@irdl_op_definition
class StoreOp(IRDLOperation, SimpleSMTLibOp):
    name = "smt.array.store"

    traits = traits_def(Pure())

    _DOMAIN: ClassVar = VarConstraint("_DOMAIN", AnyAttr())
    _RANGE: ClassVar = VarConstraint("_RANGE", AnyAttr())
    _ARRAY: ClassVar = VarConstraint("_ARRAY", ArrayType.constr(_DOMAIN, _RANGE))

    array: Operand = operand_def(_ARRAY)
    index: Operand = operand_def(_DOMAIN)
    value: Operand = operand_def(_RANGE)

    res: OpResult = result_def(_ARRAY)

    def __init__(self, array: SSAValue, index: SSAValue, value: SSAValue):
        if not isa(array.type, AnyArrayType):
            raise ValueError(f"Expected array type, got {array.type} for array operand")
        super().__init__(result_types=[array.type], operands=[array, index, value])

    def op_name(self) -> str:
        return "store"

    # assembly_format = "$array `[` $index `]` `,` $value attr-dict `:` type($array)"

    def print(self, printer: Printer):
        printer.print(" ", self.array, "[", self.index, "]", ", ", self.value)
        if self.attributes:
            printer.print_attr_dict(self.attributes)
        printer.print(" : ", self.array.type)

    @classmethod
    def parse(cls, parser: Parser) -> StoreOp:
        array = parser.parse_unresolved_operand()
        parser.parse_characters("[")
        index = parser.parse_unresolved_operand()
        parser.parse_characters("]")
        parser.parse_characters(",")
        value = parser.parse_unresolved_operand()
        attributes = parser.parse_optional_attr_dict()
        parser.parse_characters(":")
        array_type = parser.parse_type()
        if not isa(array_type, ArrayType):
            raise ValueError(f"Expected array type, got {array_type} for array operand")

        (array, index, value) = parser.resolve_operands(
            [array, index, value],
            [array_type, array_type.domain, array_type.range],
            parser.pos,
        )
        op = cls(array, index, value)
        op.attributes = attributes
        return op


SMTArray = Dialect("smt.array", [SelectOp, StoreOp], [ArrayType])
