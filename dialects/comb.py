from typing import Annotated
from xdsl.dialects.builtin import IndexType, IntegerAttr, IntegerType, UnitAttr, i32
from xdsl.ir import Dialect, OpResult
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    OpAttr,
    Operand,
    OptOpAttr,
    VarOperand,
    irdl_op_definition,
)


class BinCombOp(IRDLOperation):
    """
    A binary comb operation. It has two operands and one
    result, all of the same integer type.
    """

    T = Annotated[IntegerType, ConstraintVar("T")]

    lhs: Annotated[Operand, T]
    rhs: Annotated[Operand, T]
    result: Annotated[OpResult, T]

    two_state: OptOpAttr[UnitAttr]


class VariadicCombOp(IRDLOperation):
    """
    A variadic comb operation. It has a variadic number of operands, and a single
    result, all of the same type.
    """

    T = Annotated[IntegerType, ConstraintVar("T")]

    inputs: Annotated[VarOperand, T]
    result: Annotated[OpResult, T]

    two_state: OptOpAttr[UnitAttr]


@irdl_op_definition
class AddOp(VariadicCombOp):
    """Addition"""

    name = "comb.add"


@irdl_op_definition
class MulOp(VariadicCombOp):
    """Multiplication"""

    name = "comb.mul"


@irdl_op_definition
class DivUOp(BinCombOp):
    """Unsigned division"""

    name = "comb.divu"


@irdl_op_definition
class DivSOp(BinCombOp):
    """Signed division"""

    name = "comb.divs"


@irdl_op_definition
class ModUOp(BinCombOp):
    """Unsigned remainder"""

    name = "comb.modu"


@irdl_op_definition
class ModSOp(BinCombOp):
    """Signed remainder"""

    name = "comb.mods"


@irdl_op_definition
class ShlOp(BinCombOp):
    """Left shift"""

    name = "comb.shl"


@irdl_op_definition
class ShrUOp(BinCombOp):
    """Unsigned right shift"""

    name = "comb.shru"


@irdl_op_definition
class ShrSOp(BinCombOp):
    """Signed right shift"""

    name = "comb.shrs"


@irdl_op_definition
class SubOp(BinCombOp):
    """Subtraction"""

    name = "comb.sub"


@irdl_op_definition
class AndOp(VariadicCombOp):
    """Bitwise and"""

    name = "comb.and"


@irdl_op_definition
class OrOp(VariadicCombOp):
    """Bitwise or"""

    name = "comb.or"


@irdl_op_definition
class XorOp(VariadicCombOp):
    """Bitwise xor"""

    name = "comb.xor"


@irdl_op_definition
class ICmpOp(IRDLOperation):
    """Integer comparison"""

    T = Annotated[IntegerType, ConstraintVar("T")]

    lhs: Annotated[Operand, T]
    rhs: Annotated[Operand, T]
    result: Annotated[OpResult, IntegerType(1)]

    predicate: OpAttr[IntegerAttr[IndexType]]  # TODO: enum
    two_state: OpAttr[UnitAttr]


@irdl_op_definition
class ParityOp(IRDLOperation):
    """Parity"""

    input: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType(1)]

    two_state: OptOpAttr[UnitAttr]


@irdl_op_definition
class ExtractOp(IRDLOperation):
    """
    Extract a range of bits into a smaller value, low_bit
    specifies the lowest bit included.
    """

    input: Annotated[Operand, IntegerType]
    low_bit: OpAttr[IntegerAttr[Annotated[IntegerType, i32]]]
    result: Annotated[OpResult, IntegerType]


@irdl_op_definition
class ConcatOp(IRDLOperation):
    """
    Concatenate a variadic list of operands together.
    """

    inputs: Annotated[VarOperand, IntegerType]
    result: Annotated[OpResult, IntegerType]


@irdl_op_definition
class ReplicateOp(IRDLOperation):
    """
    Concatenate the operand a constant number of times.
    """

    input: Annotated[Operand, IntegerType]
    result: Annotated[OpResult, IntegerType]


@irdl_op_definition
class MuxOp(IRDLOperation):
    """
    Select between two values based on a condition.
    """

    T = Annotated[IntegerType, ConstraintVar("T")]

    cond: Annotated[Operand, IntegerType(1)]
    true_value: Annotated[Operand, T]
    false_value: Annotated[Operand, T]
    result: Annotated[OpResult, T]


Comb = Dialect(
    [
        AddOp,
        MulOp,
        DivUOp,
        DivSOp,
        ModUOp,
        ModSOp,
        ShlOp,
        ShrUOp,
        ShrSOp,
        SubOp,
        AndOp,
        OrOp,
        XorOp,
        ICmpOp,
        ParityOp,
        ExtractOp,
        ConcatOp,
        ReplicateOp,
        MuxOp,
    ]
)
