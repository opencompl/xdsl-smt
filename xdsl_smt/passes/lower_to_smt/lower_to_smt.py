"""
Lowers dialects to the SMT dialect
This pass can be extended with additional `RewritePattern`s to
handle more dialects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import ClassVar

from xdsl.passes import ModulePass
from xdsl.ir import Attribute, Operation, SSAValue, Region
from xdsl.traits import IsTerminator
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.pattern_rewriter import PatternRewriter

from xdsl_smt.dialects import smt_dialect
from xdsl_smt.dialects.effects.effect import StateType
from xdsl_smt.semantics.pdl_semantics import PDLSemantics
from xdsl_smt.semantics.semantics import (
    AttributeSemantics,
    OperationSemantics,
    TypeSemantics,
)

from xdsl_smt.dialects.smt_utils_dialect import PairType
from xdsl.dialects import pdl


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
    type_lowerers: ClassVar[dict[type[Attribute], TypeSemantics]] = {}
    attribute_semantics: ClassVar[dict[type[Attribute], AttributeSemantics]] = {}
    dynamic_semantics_enabled: ClassVar[bool] = False

    @staticmethod
    def lower_region(
        region: Region, effect_state: SSAValue | None
    ) -> tuple[tuple[SSAValue, ...], SSAValue | None]:
        if len(region.blocks) != 1:
            raise Exception(
                f"SMT Lowering can only lower regions with exactly one block"
            )

        # Lower the block arguments
        # Do not modify effect states, as they are still referenced by effect_state.
        for i, arg in enumerate(region.block.args):
            if isinstance(arg.type, StateType):
                continue
            new_type = SMTLowerer.lower_type(arg.type)
            new_arg = region.block.insert_arg(new_type, i)
            arg.replace_by(new_arg)
            region.block.erase_arg(arg)

        # Lower the operations
        for op in list(region.block.ops):
            effect_state = SMTLowerer.lower_operation(op, effect_state)

        # Terminators are not lowered yet, they are all considered to be yields.
        if (terminator := region.block.last_op) and terminator.has_trait(IsTerminator):
            return (tuple(terminator.operands), effect_state)
        return ((), effect_state)

    @staticmethod
    def lower_operation(
        op: Operation, effect_state: SSAValue | None
    ) -> SSAValue | None:
        if isinstance(op, pdl.PatternOp) and SMTLowerer.dynamic_semantics_enabled:
            return effect_state
        if type(op) in SMTLowerer.rewrite_patterns:
            rewriter = PatternRewriter(op)
            return SMTLowerer.rewrite_patterns[type(op)].rewrite(
                op, effect_state, rewriter, SMTLowerer()
            )

        if type(op) in SMTLowerer.op_semantics:
            rewriter = PatternRewriter(op)
            new_res, effect_state = SMTLowerer.op_semantics[type(op)].get_semantics(
                op.operands,
                op.result_types,
                {**op.attributes, **op.properties},
                effect_state,
                rewriter,
            )

            # When the semantics are PDL-based, the replacement is performed in PDL
            if not isinstance(SMTLowerer.op_semantics[type(op)], PDLSemantics):
                rewriter.replace_matched_op([], new_res)
            return effect_state

        if type(op) in smt_dialect.SMTDialect.operations:
            return effect_state

        raise Exception(f"No SMT lowering defined for the '{op.name}' operation")

    @staticmethod
    def lower_type(type_: Attribute) -> Attribute:
        """Convert a type to an SMT sort"""

        # Do not lower effect states to SMT, these are done in separate passes.
        if isinstance(type_, StateType):
            return type_
        if type(type_) not in SMTLowerer.type_lowerers:
            raise ValueError(f"Cannot lower {type_.name} type to SMT")
        return SMTLowerer.type_lowerers[type(type_)].get_semantics(type_)

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


@dataclass(frozen=True)
class LowerToSMTPass(ModulePass):
    name = "lower-to-smt"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        lowerer = SMTLowerer()
        lowerer.lower_region(op.body, None)
