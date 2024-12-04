from typing import TypeVar, IO, Callable, Sequence
from ..traits.effects import Pure
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
from xdsl.ir import SSAValue, Operation, Region, Block
from xdsl_smt.dialects import smt_dialect as smt
from xdsl.dialects.builtin import Signedness

LARGE_ENOUGH_INT_TYPE = IntegerType(128, Signedness.UNSIGNED)

_OpT = TypeVar("_OpT", bound=Operation)


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
    res: OpResult = result_def(smt.BoolType)
    lhs: Operand = operand_def(SMTIntType)
    rhs: Operand = operand_def(SMTIntType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[smt.BoolType()], operands=[lhs, rhs])

    @classmethod
    def get(cls: type[_BPOpT], lhs: SSAValue, rhs: SSAValue) -> _BPOpT:
        return cls.create(result_types=[smt.BoolType()], operands=[lhs, rhs])


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

    def __init__(self, value: int | IntegerAttr[IntegerType]):
        assert isinstance(value, int)
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


def bitwise_function(
    combine_bits: Callable[[SSAValue, SSAValue], Sequence[Operation]], name: str
):
    # The body of the function
    block = Block(arg_types=[SMTIntType(), SMTIntType(), SMTIntType()])
    k = block.args[0]
    x = block.args[1]
    y = block.args[2]
    # Constants
    one_op = ConstantOp(1)
    two_op = ConstantOp(2)
    block.add_ops([one_op, two_op])
    # Combine bits
    x_bit_op = ModOp(x, two_op.res)
    y_bit_op = ModOp(y, two_op.res)
    block.add_ops([x_bit_op, y_bit_op])
    bits_ops = combine_bits(x_bit_op.res, y_bit_op.res)
    assert bits_ops
    block.add_ops(bits_ops)
    # Recursive call
    new_x_op = DivOp(x, two_op.res)
    new_y_op = DivOp(y, two_op.res)
    k_minus_one = SubOp(k, one_op.res)
    rec_call_op = smt.RecCallOp(
        args=[k_minus_one.res, new_x_op.res, new_y_op.res],
        result_types=[SMTIntType()],
    )
    mul_op = MulOp(two_op.res, rec_call_op.res[0])
    # Result
    assert isinstance(bits_ops[-1], BinaryIntOp)
    res_op = AddOp(bits_ops[-1].res, mul_op.res)
    return_op = smt.ReturnOp(res_op.res)
    # Build the function
    block.add_ops(
        [
            new_x_op,
            new_y_op,
            k_minus_one,
            rec_call_op,
            mul_op,
            res_op,
            return_op,
        ]
    )
    region = Region([block])
    define_fun_op = smt.DefineRecFunOp(region, name)
    return define_fun_op
