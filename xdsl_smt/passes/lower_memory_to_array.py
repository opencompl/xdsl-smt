from xdsl.passes import ModulePass
from xdsl.context import MLContext

from xdsl.ir import Attribute, Operation
from xdsl.dialects.builtin import ModuleOp
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)

from xdsl_smt.dialects import (
    smt_dialect as smt,
    smt_utils_dialect as smt_utils,
    smt_int_dialect as smt_int,
    memory_dialect as mem,
    smt_array_dialect as smt_array,
)
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType

byte_type = smt_utils.PairType(BitVectorType(8), smt.BoolType())
bytes_type = smt_array.ArrayType(smt_int.SMTIntType(), byte_type)
memory_block_type = smt_utils.PairType(bytes_type, smt.BoolType())
block_id_type = smt_int.SMTIntType()
memory_type = smt_array.ArrayType(block_id_type, memory_block_type)


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


class LowerMemoryToArrayPass(ModulePass):
    name = "lower-memory-to-array"

    def apply(self, ctx: MLContext, op: ModuleOp):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerGenericOp(),
                ]
            )
        )
        walker.rewrite_module(op)
