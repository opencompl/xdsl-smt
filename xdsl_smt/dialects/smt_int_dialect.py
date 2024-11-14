from typing import TypeVar, IO
from ..traits.effects import Pure
from xdsl.dialects.builtin import Signedness
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
)
from xdsl.irdl import (
    attr_def,
    operand_def,
    result_def,
    irdl_attr_definition,
    irdl_op_definition,
    Operand,
    IRDLOperation,
)
from xdsl.ir import (
    Dialect,
    OpResult,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
)
from ..traits.smt_printer import (
    SMTLibOp,
    SimpleSMTLibOp,
    SMTLibSort,
    SMTConversionCtx,
)
from .smt_dialect import BoolType

_OpT = TypeVar("_OpT", bound=Operation)

LARGE_ENOUGH_INT_TYPE = IntegerType(128, Signedness.UNSIGNED)


@irdl_attr_definition
class SMTIntType(ParametrizedAttribute, SMTLibSort, TypeAttribute):
    name = "smt.int.int"

    def print_sort_to_smtlib(self, stream: IO[str]) -> None:
        print("Int", file=stream, end="")


_BOpT = TypeVar("_BOpT", bound="BinaryIntOp")


class BinaryIntOp(IRDLOperation, Pure):
    """Base class for binary integer operations."""

    res: OpResult = result_def(SMTIntType)
    lhs: Operand = operand_def(SMTIntType)
    rhs: Operand = operand_def(SMTIntType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[SMTIntType([])], operands=[lhs, rhs])


_BPOpT = TypeVar("_BPOpT", bound="BinaryPredIntOp")


class BinaryPredIntOp(IRDLOperation, Pure):
    res: OpResult = result_def(BoolType)
    lhs: Operand = operand_def(SMTIntType)
    rhs: Operand = operand_def(SMTIntType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[BoolType()], operands=[lhs, rhs])

    @classmethod
    def get(cls: type[_BPOpT], lhs: SSAValue, rhs: SSAValue) -> _BPOpT:
        return cls.create(result_types=[BoolType()], operands=[lhs, rhs])


_UOpT = TypeVar("_UOpT", bound="UnaryIntOp")


class UnaryIntOp(IRDLOperation, Pure):
    res: OpResult = result_def(SMTIntType)
    arg: Operand = operand_def(SMTIntType)

    def __init__(self, arg: SSAValue):
        super().__init__(result_types=[arg.type], operands=[arg])

    @classmethod
    def get(cls: type[_UOpT], arg: SSAValue) -> _UOpT:
        return cls.create(result_types=[arg.type], operands=[arg])


@irdl_op_definition
class ConstantOp(IRDLOperation, Pure, SMTLibOp):
    name = "smt.int.constant"
    res: OpResult = result_def(SMTIntType)
    value: IntegerAttr[IntegerType] = attr_def(IntegerAttr)

    def __init__(self, value: int):
        super().__init__(
            result_types=[SMTIntType()],
            attributes={"value": IntegerAttr(value, LARGE_ENOUGH_INT_TYPE)},
        )

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        print(self.value.value.data, file=stream, end="")


@irdl_op_definition
class AddOp(BinaryIntOp, SimpleSMTLibOp):
    name = "smt.int.add"

    def op_name(self) -> str:
        return "+"


@irdl_op_definition
class SubOp(BinaryIntOp, SimpleSMTLibOp):
    name = "smt.int.sub"

    def op_name(self) -> str:
        return "-"


@irdl_op_definition
class NegOp(UnaryIntOp, SimpleSMTLibOp):
    name = "smt.int.neg"

    def op_name(self) -> str:
        return "-"


@irdl_op_definition
class MulOp(BinaryIntOp, SimpleSMTLibOp):
    name = "smt.int.mul"

    def op_name(self) -> str:
        return "*"


@irdl_op_definition
class DivOp(BinaryIntOp, SimpleSMTLibOp):
    name = "smt.int.div"

    def op_name(self) -> str:
        return "div"


@irdl_op_definition
class ModOp(BinaryIntOp, SimpleSMTLibOp):
    name = "smt.int.mod"

    def op_name(self) -> str:
        return "mod"


@irdl_op_definition
class AbsOp(UnaryIntOp, SimpleSMTLibOp):
    name = "smt.int.abs"

    def op_name(self) -> str:
        return "abs"


@irdl_op_definition
class LeOp(BinaryPredIntOp, SimpleSMTLibOp):
    name = "smt.int.le"

    def op_name(self) -> str:
        return "<="


@irdl_op_definition
class LtOp(BinaryPredIntOp, SimpleSMTLibOp):
    name = "smt.int.lt"

    def op_name(self) -> str:
        return "<"


@irdl_op_definition
class GeOp(BinaryPredIntOp, SimpleSMTLibOp):
    name = "smt.int.ge"

    def op_name(self) -> str:
        return ">="


@irdl_op_definition
class GtOp(BinaryPredIntOp, SimpleSMTLibOp):
    name = "smt.int.gt"

    def op_name(self) -> str:
        return ">"


SMTIntDialect = Dialect(
    "smt.int",
    [
        ConstantOp,
        AddOp,
        SubOp,
        NegOp,
        MulOp,
        DivOp,
        ModOp,
        AbsOp,
        LeOp,
        LtOp,
        GeOp,
        GtOp,
    ],
    [SMTIntType],
)
