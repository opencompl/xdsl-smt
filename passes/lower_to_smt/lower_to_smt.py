"""
Lowers dialects to the SMT dialect
This pass can be extended with additional RewritePattern to
handle more dialects.
"""

from xdsl.passes import ModulePass
from xdsl.ir import Attribute, MLContext
from xdsl.dialects.builtin import IntegerType, ModuleOp
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker


from .arith_to_smt import arith_to_smt_patterns
from .comb_to_smt import comb_to_smt_patterns
from .transfer_to_smt import transfer_to_smt_patterns
from dialects.smt_bitvector_dialect import BitVectorType


def convert_type(type: Attribute) -> Attribute:
    """Convert a type to an SMT sort"""
    if isinstance(type, IntegerType):
        return BitVectorType(type.width)
    raise Exception(f"Cannot convert {type} attribute")


class LowerToSMT(ModulePass):
    name = "lower-to-smt"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    *arith_to_smt_patterns,
                    *comb_to_smt_patterns,
                    *transfer_to_smt_patterns,
                ]
            )
        )
        walker.rewrite_module(op)
