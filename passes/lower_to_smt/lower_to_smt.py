"""
Lowers dialects to the SMT dialect
This pass can be extended with additional RewritePattern to
handle more dialects.
"""
from dataclasses import dataclass
from typing import ClassVar

from xdsl.passes import ModulePass
from xdsl.ir import Attribute, MLContext
from xdsl.dialects.builtin import IntegerType, ModuleOp
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    RewritePattern,
)

from dialects.smt_bitvector_dialect import BitVectorType


def convert_type(type: Attribute) -> Attribute:
    """Convert a type to an SMT sort"""
    if isinstance(type, IntegerType):
        return BitVectorType(type.width)
    raise Exception(f"Cannot convert {type} attribute")


@dataclass
class LowerToSMT(ModulePass):
    name = "lower-to-smt"

    rewrite_patterns: ClassVar[list[RewritePattern]] = []

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(self.rewrite_patterns)
        )
        walker.rewrite_module(op)
