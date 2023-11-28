from abc import abstractmethod
from typing import Generic, Mapping, Sequence

from xdsl.ir import Attribute, AttributeCovT, AttributeInvT, Region, SSAValue
from xdsl.pattern_rewriter import PatternRewriter


class OperationSemantics:
    """
    The semantics of an operation represented as a rewrite into the SMT dialect.
    As semantics might be used in a meta-level such as PDL, attributes are passed
    as a sequence of SSA values rather than attributes.
    Regions are expected to be treated as pure regions of code, and their semantics
    are handled outside of this function. They should generally just be moved, instead
    of modified.
    The return SSA values represent the semantics of the operation results.
    """

    @abstractmethod
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        regions: Sequence[Region],
        attributes: Mapping[str, Attribute | SSAValue],
        rewriter: PatternRewriter,
    ) -> Sequence[SSAValue]:
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
    def get_semantics(self, attribute: Attribute) -> Attribute:
        pass
