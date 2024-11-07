from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.context import MLContext

from xdsl.ir import Attribute, Operation, SSAValue, ParametrizedAttribute
from xdsl.dialects.builtin import ModuleOp
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.utils.isattr import isattr

from xdsl_smt.dialects import (
    smt_dialect as smt,
    smt_utils_dialect as smt_utils,
    smt_int_dialect as smt_int,
    memory_dialect as mem,
    smt_array_dialect as smt_array,
    smt_bitvector_dialect as smt_bv,
)
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType

byte_type = smt_utils.PairType(BitVectorType(8), smt.BoolType())
bytes_type = smt_array.ArrayType(smt_bv.BitVectorType(64), byte_type)
memory_block_type = smt_utils.PairType(
    bytes_type, smt_utils.PairType(smt_bv.BitVectorType(64), smt.BoolType())
)
block_id_type = smt_int.SMTIntType()
memory_type = smt_array.ArrayType(block_id_type, memory_block_type)


@dataclass
class MemoryBlockValueAdaptor:
    """
    Adaptor for a memory block value.
    Contains accessors for the memory block's components.
    """

    bytes: SSAValue
    size: SSAValue
    live_marker: SSAValue

    @staticmethod
    def from_value(value: SSAValue, rewriter: PatternRewriter):
        access_bytes = smt_utils.FirstOp(value)
        access_others = smt_utils.SecondOp(value)
        access_size = smt_utils.FirstOp(access_others.res)
        access_live_marker = smt_utils.SecondOp(access_others.res)

        rewriter.insert_op_before_matched_op(
            [access_bytes, access_others, access_size, access_live_marker]
        )
        return MemoryBlockValueAdaptor(
            access_bytes.res, access_size.res, access_live_marker.res
        )

    def merge_into_pairs(self, rewriter: PatternRewriter):
        other_pairs = smt_utils.PairOp(self.size, self.live_marker)
        top_pairs = smt_utils.PairOp(self.bytes, other_pairs.res)
        rewriter.insert_op_before_matched_op([other_pairs, top_pairs])
        return top_pairs.res


def recursively_convert_attr(attr: Attribute) -> Attribute:
    """
    Recursively convert an attribute to replace all references to the memory types into
    the corresponding smt types.
    """
    if isinstance(attr, mem.MemoryType):
        return memory_type
    elif isinstance(attr, mem.BlockIDType):
        return block_id_type
    elif isinstance(attr, mem.MemoryBlockType):
        return memory_block_type
    elif isinstance(attr, mem.BytesType):
        return bytes_type

    if isinstance(attr, ParametrizedAttribute):
        return attr.new(
            [recursively_convert_attr(sub_attr) for sub_attr in attr.parameters]
        )
    return attr


class LowerGenericOp(RewritePattern):
    """
    Recursively lower all result types, attributes, and properties that reference
    memory types.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        for result in op.results:
            if (new_type := recursively_convert_attr(result.type)) != result.type:
                rewriter.modify_value_type(result, new_type)

        for region in op.regions:
            for block in region.blocks:
                for arg in block.args:
                    if (new_type := recursively_convert_attr(arg.type)) != arg.type:
                        rewriter.modify_value_type(arg, new_type)

        has_done_action = False
        for name, attr in op.attributes.items():
            if (new_attr := recursively_convert_attr(attr)) != attr:
                op.attributes[name] = new_attr
                has_done_action = True
        for name, attr in op.properties.items():
            if (new_attr := recursively_convert_attr(attr)) != attr:
                op.properties[name] = new_attr
                has_done_action = True
        if has_done_action:
            rewriter.handle_operation_modification(op)


class GetBlockPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.GetBlockOp, rewriter: PatternRewriter):
        select_op = smt_array.SelectOp(op.memory, op.block_id)
        rewriter.replace_matched_op(select_op)


class SetBlockPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.SetBlockOp, rewriter: PatternRewriter):
        store_op = smt_array.StoreOp(op.memory, op.block_id, op.block)
        rewriter.replace_matched_op(store_op)


class GetBlockBytesPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.GetBlockBytesOp, rewriter: PatternRewriter):
        block_adaptor = MemoryBlockValueAdaptor.from_value(op.memory_block, rewriter)
        rewriter.replace_matched_op([], [block_adaptor.bytes])


class SetBlockBytesPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.SetBlockBytesOp, rewriter: PatternRewriter):
        block_adaptor = MemoryBlockValueAdaptor.from_value(op.memory_block, rewriter)
        block_adaptor.bytes = op.bytes
        block = block_adaptor.merge_into_pairs(rewriter)
        rewriter.replace_matched_op([], [block])


class GetBlockSizePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.GetBlockSizeOp, rewriter: PatternRewriter):
        block_adaptor = MemoryBlockValueAdaptor.from_value(op.memory_block, rewriter)
        rewriter.replace_matched_op([], [block_adaptor.size])


class SetBlockSizePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.SetBlockSizeOp, rewriter: PatternRewriter):
        block_adaptor = MemoryBlockValueAdaptor.from_value(op.memory_block, rewriter)
        block_adaptor.size = op.size
        block = block_adaptor.merge_into_pairs(rewriter)
        rewriter.replace_matched_op([], [block])


class GetBlockIsLivePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: mem.GetBlockLiveMarkerOp, rewriter: PatternRewriter
    ):
        block_adaptor = MemoryBlockValueAdaptor.from_value(op.memory_block, rewriter)
        rewriter.replace_matched_op([], [block_adaptor.live_marker])


class SetBlockIsLivePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: mem.SetBlockLiveMarkerOp, rewriter: PatternRewriter
    ):
        block_adaptor = MemoryBlockValueAdaptor.from_value(op.memory_block, rewriter)
        block_adaptor.live_marker = op.live
        block = block_adaptor.merge_into_pairs(rewriter)
        rewriter.replace_matched_op([], [block])


class ReadBytesPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.ReadBytesOp, rewriter: PatternRewriter):
        assert isattr(
            res_type := op.res.type, smt_utils.PairType[BitVectorType, smt.BoolType]
        )
        if res_type.first.width.data != 8:
            raise NotImplementedError(
                f"Only 8-bit reads are supported for {mem.ReadBytesOp.name} in "
                "its lowering to SMT arrays"
            )
        select_op = smt_array.SelectOp(op.bytes, op.index)
        rewriter.replace_matched_op(select_op)


class WriteBytesPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.WriteBytesOp, rewriter: PatternRewriter):
        assert isattr(
            val_type := op.value.type, smt_utils.PairType[BitVectorType, smt.BoolType]
        )
        if val_type.first.width.data != 8:
            raise NotImplementedError(
                f"Only 8-bit writes are supported for {mem.WriteBytesOp.name} in "
                "its lowering to SMT arrays"
            )
        store_op = smt_array.StoreOp(op.bytes, op.index, op.value)
        rewriter.replace_matched_op(store_op)


@dataclass
class GetFreshBlockIDPattern(RewritePattern):
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: mem.GetFreshBlockIDOp, rewriter: PatternRewriter):
        # Get a new fresh value for the block ID
        fresh_id_op = smt_int.ConstantOp(self.counter)
        rewriter.insert_op_before_matched_op([fresh_id_op])
        self.counter += 1

        # Replace the original operation
        rewriter.replace_matched_op([], [fresh_id_op.res, op.memory])


class LowerMemoryToArrayPass(ModulePass):
    name = "lower-memory-to-array"

    def apply(self, ctx: MLContext, op: ModuleOp):
        for sub_op in op.body.ops:
            if isinstance(sub_op, smt.DefineFunOp):
                raise Exception(
                    "Cannot lower memory operations when functions are present. "
                    "Please run the 'inline-functions' pass first."
                )

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerGenericOp(),
                    GetBlockPattern(),
                    SetBlockPattern(),
                    GetBlockBytesPattern(),
                    SetBlockBytesPattern(),
                    GetBlockSizePattern(),
                    SetBlockSizePattern(),
                    GetBlockIsLivePattern(),
                    SetBlockIsLivePattern(),
                    ReadBytesPattern(),
                    WriteBytesPattern(),
                    GetFreshBlockIDPattern(),
                ]
            )
        )
        walker.rewrite_module(op)
