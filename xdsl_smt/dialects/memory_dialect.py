from xdsl.ir import ParametrizedAttribute, TypeAttribute, SSAValue, Dialect
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    operand_def,
    result_def,
    irdl_op_definition,
)

from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType
from xdsl_smt.dialects.smt_dialect import BoolType


@irdl_attr_definition
class MemoryType(ParametrizedAttribute, TypeAttribute):
    """Type of the memory state."""

    name = "memory.memory"


@irdl_attr_definition
class BlockIDType(ParametrizedAttribute, TypeAttribute):
    """Type of a block ID."""

    name = "memory.block_id"


@irdl_attr_definition
class MemoryBlockType(ParametrizedAttribute, TypeAttribute):
    """Type of a memory block."""

    name = "memory.block"


@irdl_attr_definition
class ByteType(ParametrizedAttribute, TypeAttribute):
    """Type of a memory byte."""

    name = "memory.byte"


@irdl_op_definition
class GetBlockOp(IRDLOperation):
    """Get a memory block from a memory state."""

    name = "memory.get_block"

    memory = operand_def(MemoryType())
    block_id = operand_def(BlockIDType())

    res = result_def(MemoryBlockType())

    assembly_format = "$memory `[` $block_id `]` attr-dict"

    def __init__(self, memory: SSAValue, block_id: SSAValue):
        super().__init__(operands=[memory, block_id], result_types=[MemoryBlockType()])
        self.res.name_hint = "block"


@irdl_op_definition
class SetBlockOp(IRDLOperation):
    """Set a memory block in a memory state."""

    name = "memory.set_block"

    memory = operand_def(MemoryType())
    block_id = operand_def(BlockIDType())
    block = operand_def(MemoryBlockType())

    res = result_def(MemoryType())

    assembly_format = "$memory `[` $block_id `]` `,` $block attr-dict"

    def __init__(self, memory: SSAValue, block_id: SSAValue, block: SSAValue):
        super().__init__(
            operands=[memory, block_id, block], result_types=[MemoryType()]
        )
        self.res.name_hint = "memory"


@irdl_op_definition
class GetBlockSizeOp(IRDLOperation):
    """Get the size of a memory block in bytes."""

    name = "memory.get_block_size"

    memory_block = operand_def(MemoryBlockType())

    res = result_def(BitVectorType(64))

    assembly_format = "$memory_block attr-dict"

    def __init__(self, memory_block: SSAValue):
        super().__init__(operands=[memory_block], result_types=[BitVectorType(64)])
        self.res.name_hint = "block_size"


@irdl_op_definition
class SetBlockSizeOp(IRDLOperation):
    """Set the size of a memory block in bytes."""

    name = "memory.set_block_size"

    memory_block = operand_def(MemoryBlockType())
    size = operand_def(BitVectorType(64))

    res = result_def(MemoryBlockType())

    assembly_format = "$memory_block `,` $size attr-dict"

    def __init__(self, memory_block: SSAValue, size: SSAValue):
        super().__init__(
            operands=[memory_block, size], result_types=[MemoryBlockType()]
        )
        self.res.name_hint = "block"


@irdl_op_definition
class GetBlockLiveMarkerOp(IRDLOperation):
    """
    Get the live marker of a memory block.
    A block is live if it is reachable from the memory state.
    """

    name = "memory.get_live_marker"

    memory_block = operand_def(MemoryBlockType())

    res = result_def(BoolType())

    assembly_format = "$memory_block attr-dict"

    def __init__(self, memory_block: SSAValue):
        super().__init__(operands=[memory_block], result_types=[BoolType()])
        self.res.name_hint = "is_live"


@irdl_op_definition
class SetBlockLiveMarkerOp(IRDLOperation):
    """
    Set the live marker of a memory block.
    A block is live if it is reachable from the memory state.
    """

    name = "memory.set_block_live_marker"

    memory_block = operand_def(MemoryBlockType())
    live = operand_def(BoolType())

    res = result_def(MemoryBlockType())

    assembly_format = "$memory_block `,` $live attr-dict"

    def __init__(self, memory_block: SSAValue, live: SSAValue):
        super().__init__(
            operands=[memory_block, live], result_types=[MemoryBlockType()]
        )
        self.res.name_hint = "block"


@irdl_op_definition
class GetFreshBlockIDOp(IRDLOperation):
    """
    Allocate a fresh block ID.
    The block ID is different than any block that is currently live.
    In particular, it may reuse block IDs of blocks that are no longer live.
    """

    name = "memory.get_fresh_block_id"

    memory = operand_def(MemoryType())

    res = result_def(BlockIDType())

    assembly_format = "$memory attr-dict"

    def __init__(self, memory: SSAValue):
        super().__init__(operands=[memory], result_types=[BlockIDType()])
        self.res.name_hint = "bid"


MemoryDialect = Dialect(
    "memory",
    [
        GetBlockOp,
        SetBlockOp,
        GetBlockSizeOp,
        SetBlockSizeOp,
        GetBlockLiveMarkerOp,
        SetBlockLiveMarkerOp,
        GetFreshBlockIDOp,
    ],
    [
        MemoryType,
        BlockIDType,
        MemoryBlockType,
        ByteType,
    ],
)
