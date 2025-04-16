"""
Define the data structures to represent semantics of operations, attributes, and types.
These structures are used to manipulate the semantics of operations when attributes are
also defined as ssa values.
Otherwise, semantics should be defined using the `SMTRewritePattern` class.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Mapping, Sequence

from xdsl.ir import Attribute, SSAValue
from xdsl.pattern_rewriter import PatternRewriter


class OperationSemantics:
    """
    The semantics of an operation represented as a rewrite into the SMT dialect.
    """

    @abstractmethod
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        """
        As semantics might be used in a meta-level such as PDL, attributes are passed
        as a sequence of SSA values rather than attributes.
        Regions are expected to be treated as pure regions of code, and their semantics
        are handled outside of this function. They should generally just be moved, instead
        of modified.
        The sequence of SSA values represent the semantics of the operation results,
        and the SSA value represents the new effect state.
        """
        pass


class AttributeSemantics:
    """
    The semantics of a class of attributes.
    When an attribute or property has a meaning in an operation,
    it should be first lowered into an SSAValue that encodes that meaning.
    """

    @abstractmethod
    def get_semantics(
        self,
        attribute: Attribute,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        pass


class TypeSemantics:
    """
    The semantics of a class of types.
    Returns an Attribute that encodes the type as an SMT sort.
    """

    @abstractmethod
    def get_semantics(self, type: Attribute) -> Attribute:
        pass


class RefinementSemantics:
    """
    The semantics of a refinement.
    Returns a boolean SSAValue that is true if the refinement holds.
    """

    @abstractmethod
    def get_semantics(
        self,
        val_before: SSAValue,
        val_after: SSAValue,
        state_before: SSAValue,
        state_after: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        pass
