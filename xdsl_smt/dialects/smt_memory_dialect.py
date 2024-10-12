from xdsl.ir import TypeAttribute, ParametrizedAttribute, Dialect
from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
    IRDLOperation,
)
from xdsl.utils.exceptions import VerifyException
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType
from xdsl_smt.semantics.semantics import EffectState


@irdl_attr_definition
class MemoryStateType(TypeAttribute, ParametrizedAttribute, EffectState):
    """
    Type of a memory effect state.
    The memory effect state contains the memory state of the program.
    The only operations that can currently be performed on the memory state are
    memory reads and writes.
    """

    name = "smt_mem.state"

    def __init__(self):
        super().__init__(())


@irdl_attr_definition
class PointerType(TypeAttribute, ParametrizedAttribute, EffectState):
    """
    Type of a pointer. It is used to represent an address in memory.
    """

    name = "smt_mem.ptr"

    def __init__(self):
        super().__init__(())


@irdl_op_definition
class OffsetPointerOp(IRDLOperation):
    """Offset a pointer by an integer."""

    name = "smt_mem.offset_ptr"

    pointer = operand_def(PointerType())
    offset = operand_def(BitVectorType(64))

    res = result_def(PointerType())

    assembly_format = "$pointer `[` $offset `]` attr-dict"


@irdl_op_definition
class ReadOp(IRDLOperation):
    """Read at a memory location."""

    name = "smt_mem.read"

    state = operand_def(MemoryStateType())
    pointer = operand_def(PointerType())

    res = result_def(BitVectorType)

    assembly_format = "$state `[` $pointer `]` attr-dict `:` type($res)"

    def verify_(self):
        assert isinstance(self.res.type, BitVectorType)
        size = self.res.type.width.data
        if size % 8 != 0:
            raise VerifyException("read size must be a multiple of 8")


@irdl_op_definition
class WriteOp(IRDLOperation):
    """Write at a memory location."""

    name = "smt_mem.write"

    state = operand_def(MemoryStateType())
    pointer = operand_def(PointerType())
    value = operand_def(BitVectorType)

    res = result_def(MemoryStateType())

    assembly_format = "$state `[` $pointer `]` `,` $value attr-dict `:` type($value)"

    def verify_(self):
        assert isinstance(self.value.type, BitVectorType)
        if self.value.type.width.data % 8 != 0:
            raise VerifyException("written size must be a multiple of 8")


@irdl_op_definition
class AllocOp(IRDLOperation):
    """
    Allocate memory at a given location.
    Memory is not initialized, and may contain any value.
    """

    name = "smt_mem.alloc"

    state = operand_def(MemoryStateType())
    size = operand_def(BitVectorType(64))

    pointer = result_def(PointerType())
    new_state = result_def(MemoryStateType())

    assembly_format = "$state `,` $size attr-dict"


@irdl_op_definition
class DeallocOp(IRDLOperation):
    """
    Deallocate memory at a given location.
    The memory should have been previously allocated at that location.
    """

    name = "smt_mem.dealloc"

    state = operand_def(MemoryStateType())
    pointer = operand_def(PointerType())

    new_state = result_def(MemoryStateType())

    assembly_format = "$state `,` $pointer attr-dict"


SMTMemoryDialect = Dialect(
    "smt_mem",
    [
        OffsetPointerOp,
        ReadOp,
        WriteOp,
        AllocOp,
        DeallocOp,
    ],
    [
        MemoryStateType,
        PointerType,
    ],
)
