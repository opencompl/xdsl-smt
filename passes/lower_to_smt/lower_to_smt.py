"""
Lowers dialects to the SMT dialect
This pass can be extended with additional RewritePattern to
handle more dialects.
"""
from dataclasses import dataclass
from typing import Callable, ClassVar

from xdsl.passes import ModulePass
from xdsl.ir import Attribute, MLContext
from xdsl.dialects.builtin import IntegerType, ModuleOp
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    RewritePattern,
)

from dialects.smt_bitvector_dialect import BitVectorType


def integer_type_lowerer(type: Attribute) -> Attribute | None:
    """Convert a type to an SMT sort"""
    if isinstance(type, IntegerType):
        return BitVectorType(type.width)
    return None


@dataclass
class LowerToSMT(ModulePass):
    name = "lower-to-smt"

    type_lowerers: ClassVar[list[Callable[[Attribute], Attribute | None]]] = []
    rewrite_patterns: ClassVar[list[RewritePattern]] = []

    @staticmethod
    def lower_type(type: Attribute) -> Attribute:
        for lowerer in LowerToSMT.type_lowerers:
            if res := lowerer(type):
                return res
        raise ValueError(f"Cannot lower {type} to SMT")

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(self.rewrite_patterns)
        )
        walker.rewrite_module(op)
