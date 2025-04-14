"""
A rewrite pattern that is used to represent semantics.
Compared to rewrite patterns, it expects and returns an effect state.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xdsl_smt.passes.lower_to_smt.smt_lowerer import SMTLowerer


class SMTLoweringRewritePattern(ABC):
    """
    This class represents a rewrite pattern used in an SMT lowering.
    The difference with a traditional rewrite pattern is that is needs to pass and
    return all effect states.
    """

    @abstractmethod
    def rewrite(
        self,
        op: Operation,
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
        smt_lowerer: SMTLowerer,
    ) -> SSAValue | None:
        pass
