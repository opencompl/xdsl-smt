from xdsl.utils.hints import isa
from xdsl.ir import TypeAttribute, ParametrizedAttribute, Dialect, SSAValue
from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    IRDLOperation,
)
from xdsl.utils.exceptions import VerifyException
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType
from xdsl_smt.dialects.effects.effect import StateType
from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.dialects.smt_utils_dialect import PairType


@irdl_attr_definition
class PointerType(TypeAttribute, ParametrizedAttribute):
    """
    Type of a pointer. It is used to represent an address in memory.
    """

    name = "mem_effect.ptr"

    def __init__(self):
        super().__init__(())


@irdl_op_definition
class OffsetPointerOp(IRDLOperation):
    """Offset a pointer by an integer."""

    name = "mem_effect.offset_ptr"

    pointer = operand_def(PointerType())
    offset = operand_def(BitVectorType(64))

    res = result_def(PointerType())

    assembly_format = "$pointer `[` $offset `]` attr-dict"

    def __init__(self, pointer: SSAValue, offset: SSAValue):
        super().__init__(
            operands=[pointer, offset],
            result_types=[PointerType()],
        )


@irdl_op_definition
class ReadOp(IRDLOperation):
    """Read at a memory location."""

    name = "mem_effect.read"

    state = operand_def(StateType())
    pointer = operand_def(PointerType())

    new_state = result_def(StateType())
    res = result_def(PairType[BitVectorType, BoolType])

    assembly_format = "$state `[` $pointer `]` attr-dict `:` type($res)"

    def verify_(self):
        assert isa(self.res.type, PairType[BitVectorType, BoolType])
        size = self.res.type.first.width.data
        if size % 8 != 0:
            raise VerifyException("read size must be a multiple of 8")

    def __init__(
        self,
        state: SSAValue,
        pointer: SSAValue,
        result_type: PairType[BitVectorType, BoolType],
    ):
        super().__init__(
            operands=[state, pointer],
            result_types=[StateType(), result_type],
        )


@irdl_op_definition
class WriteOp(IRDLOperation):
    """Write at a memory location."""

    name = "mem_effect.write"

    value = operand_def(BitVectorType)
    state = operand_def(StateType())
    pointer = operand_def(PointerType())

    new_state = result_def(StateType())

    assembly_format = "$value `,` $state `[` $pointer `]` attr-dict `:` type($value)"

    def verify_(self):
        assert isinstance(self.value.type, BitVectorType)
        if self.value.type.width.data % 8 != 0:
            raise VerifyException("written size must be a multiple of 8")

    def __init__(self, value: SSAValue, state: SSAValue, pointer: SSAValue):
        super().__init__(
            operands=[value, state, pointer],
            result_types=[StateType()],
        )


@irdl_op_definition
class AllocOp(IRDLOperation):
    """
    Allocate memory at a given location.
    Memory is not initialized, and may contain any value.
    """

    name = "mem_effect.alloc"

    state = operand_def(StateType())
    size = operand_def(BitVectorType(64))

    new_state = result_def(StateType())
    pointer = result_def(PointerType())

    assembly_format = "$state `,` $size attr-dict"

    def __init__(self, state: SSAValue, size: SSAValue):
        super().__init__(
            operands=[state, size],
            result_types=[StateType(), PointerType()],
        )


@irdl_op_definition
class DeallocOp(IRDLOperation):
    """
    Deallocate memory at a given location.
    The memory should have been previously allocated at that location.
    """

    name = "mem_effect.dealloc"

    state = operand_def(StateType())
    pointer = operand_def(PointerType())

    new_state = result_def(StateType())

    assembly_format = "$state `,` $pointer attr-dict"

    def __init__(self, state: SSAValue, pointer: SSAValue):
        super().__init__(
            operands=[state, pointer],
            result_types=[StateType()],
        )


MemoryEffectDialect = Dialect(
    "mem_effect",
    [
        OffsetPointerOp,
        ReadOp,
        WriteOp,
        AllocOp,
        DeallocOp,
    ],
    [
        PointerType,
    ],
)
