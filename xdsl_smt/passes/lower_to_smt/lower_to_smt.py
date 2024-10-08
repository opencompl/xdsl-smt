"""
Lowers dialects to the SMT dialect
This pass can be extended with additional `RewritePattern`s to
handle more dialects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Callable, ClassVar

from xdsl.passes import ModulePass
from xdsl.ir import Attribute, Operation, SSAValue, Region
from xdsl.traits import IsTerminator
from xdsl.context import MLContext
from xdsl.dialects.builtin import IntegerType, ModuleOp
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
)

from xdsl_smt.dialects import smt_dialect
from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.semantics.pdl_semantics import PDLSemantics
from xdsl_smt.semantics.semantics import (
    AttributeSemantics,
    EffectState,
    EffectStates,
    OperationSemantics,
)

from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType
from xdsl_smt.dialects.smt_utils_dialect import PairType
from xdsl.dialects import pdl

def integer_poison_type_lowerer(type: Attribute) -> Attribute | None:
    """Convert an integer type to a bitvector integer with a poison flag."""
    if isinstance(type, IntegerType):
        return PairType(BitVectorType(type.width), BoolType())
    return None


def integer_type_lowerer(type: Attribute) -> Attribute | None:
    """Convert an integer type to a bitvector integer."""
    if isinstance(type, IntegerType):
        return BitVectorType(type.width)
    return None


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
        effect_states: EffectStates,
        rewriter: PatternRewriter,
        smt_lowerer: SMTLowerer,
    ) -> EffectStates:
        pass


@dataclass
class SMTLowerer:
    """
    Lowers operations, regions, types, and attributes to SMT.
    Operations are lowered by calling their respective rewrite patterns.
    Regions are lowered by lowering in order all operations, mapping some
    inputs to the entry basic block, and returning the operands given to its
    terminator.
    Attributes and types are lowered by just replacing them with their SMT semantics.
    """

    rewrite_patterns: ClassVar[dict[type[Operation], SMTLoweringRewritePattern]] = {}
    op_semantics: ClassVar[dict[type[Operation], OperationSemantics]] = {}
    type_lowerers: ClassVar[list[Callable[[Attribute], Attribute | None]]] = []
    attribute_semantics: ClassVar[dict[type[Attribute], AttributeSemantics]] = {}
    effect_types: ClassVar[list[Attribute]] = []
    dynamic_semantics_enabled: ClassVar[bool] = False

    @staticmethod
    def lower_region(
        region: Region, effect_states: EffectStates
    ) -> tuple[tuple[SSAValue, ...], EffectStates]:
        if len(region.blocks) != 1:
            raise Exception(
                f"SMT Lowering can only lower regions with exactly one block"
            )

        # Lower the block arguments
        # Do not modify effect states, as they are still referenced by effect_states.
        for i, arg in enumerate(region.block.args):
            if (new_type := SMTLowerer.lower_type(arg.type)) != arg.type:
                new_arg = region.block.insert_arg(new_type, i)
                arg.replace_by(new_arg)
                region.block.erase_arg(arg)

        # Lower the operations
        for op in list(region.block.ops):
            if (
                    isinstance(op,pdl.PatternOp)
                    and SMTLowerer.dynamic_semantics_enabled
            ):
                pass
            else:
                effect_states = SMTLowerer.lower_operation(op, effect_states)

        # Terminators are not lowered yet, they are all considered to be yields.
        if (terminator := region.block.last_op) and terminator.has_trait(IsTerminator):
            return (tuple(terminator.operands), effect_states)
        return ((), effect_states)

    @staticmethod
    def lower_operation(op: Operation, effect_states: EffectStates) -> EffectStates:
        if type(op) in SMTLowerer.rewrite_patterns:
            rewriter = PatternRewriter(op)
            return SMTLowerer.rewrite_patterns[type(op)].rewrite(
                op, effect_states, rewriter, SMTLowerer()
            )

        if type(op) in SMTLowerer.op_semantics:
            rewriter = PatternRewriter(op)
            new_res, effect_states = SMTLowerer.op_semantics[type(op)].get_semantics(
                op.operands,
                op.result_types,
                {**op.attributes, **op.properties},
                effect_states,
                rewriter,
            )

            # When the semantics are PDL-based, the replacement is performed in PDL
            if not isinstance(SMTLowerer.op_semantics[type(op)],PDLSemantics):
                rewriter.replace_matched_op([], new_res)
            return effect_states

        if type(op) in smt_dialect.SMTDialect.operations:
            return effect_states

        raise Exception(f"No SMT lowering defined for the '{op.name}' operation")

    @staticmethod
    def lower_type(type: Attribute) -> Attribute:
        """Convert a type to an SMT sort"""

        # Do not lower effect states to SMT, these are done in separate passes.
        if isinstance(type, EffectState):
            return type
        for lowerer in SMTLowerer.type_lowerers:
            if res := lowerer(type):
                return res
        raise ValueError(f"Cannot lower {type} to SMT")

    @staticmethod
    def lower_types(*types: Attribute) -> Attribute:
        """Convert a list of types into a cons-list of SMT pairs"""

        if len(types) == 0:
            raise ValueError("Must have at least one type")
        elif len(types) == 1:
            return __class__.lower_type(types[0])
        else:
            return reduce(
                lambda r, l: PairType(l, r), map(__class__.lower_type, reversed(types))
            )


@dataclass
class OperationSemanticsRewritePattern(RewritePattern):
    semantics: dict[type[Operation], OperationSemantics]

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if type(op) in self.semantics:
            assert not op.regions
            res_values, _ = self.semantics[type(op)].get_semantics(
                op.operands,
                tuple(res.type for res in op.results),
                {**op.attributes, **op.properties},
                EffectStates({}),
                rewriter,
            )
            rewriter.replace_matched_op([], res_values)
        return None


@dataclass(frozen=True)
class LowerToSMTPass(ModulePass):
    name = "lower-to-smt"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        lowerer = SMTLowerer()
        lowerer.lower_region(op.body, EffectStates({}))
