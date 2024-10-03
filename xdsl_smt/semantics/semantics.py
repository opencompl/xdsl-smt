from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence

from xdsl.ir import Attribute, SSAValue
from xdsl.pattern_rewriter import PatternRewriter


class EffectState:
    """Mark an attribute as an effect state."""


@dataclass(frozen=True)
class EffectStates:
    """
    The state of all effects. It contains a dictionary associating each handled effect
    with their current state.
    """

    states: dict[Attribute, SSAValue]
    """
    A dictionary containing all effect states.
    It is indexed by the type of each effect.
    """

    def copy(self) -> EffectStates:
        """Copy the states of all effects."""
        return EffectStates(self.states.copy())

    def with_updated_effects(self, value: Mapping[Attribute, SSAValue]) -> EffectStates:
        """Get new effect states updated with the given states."""
        return EffectStates({**self.states, **value})


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
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:
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
        states_before: EffectStates,
        states_after: EffectStates,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        pass
