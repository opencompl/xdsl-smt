from __future__ import annotations

from abc import abstractmethod
from typing import Mapping, Sequence

from xdsl.ir import Attribute, SSAValue
from xdsl.builder import Builder
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

    @abstractmethod
    def get_unbounded_semantics(
        self,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        raise Exception(
            f"Cannot define the semantics of an unbounded attribute of this kind"
        )


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
        builder: Builder,
    ) -> SSAValue:
        pass
