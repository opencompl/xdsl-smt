from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Callable, Tuple
from xdsl.dialects.pdl import (
    ApplyNativeConstraintOp,
    ApplyNativeRewriteOp,
    OperationOp,
    PatternOp,
    ReplaceOp,
    RewriteOp,
)
from xdsl.ir import Operation, SSAValue
from xdsl_smt.semantics.refinements import IntegerTypeRefinementSemantics
from xdsl_smt.semantics.semantics import RefinementSemantics
from xdsl.context import MLContext
from xdsl.rewriter import InsertPoint, Rewriter
from ..dialects.smt_dialect import DeclareConstOp
from xdsl_smt.dialects.effects.effect import StateType
from xdsl_smt.semantics.arith_int_semantics import (
    IntIntegerTypeRefinementSemantics,
)
from xdsl.dialects.builtin import IntegerType, UnitAttr
from xdsl_smt.semantics.builtin_semantics import (
    IndexTypeSemantics,
    IntegerAttrSemantics,
    IntegerTypeSemantics,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType, IndexType
from xdsl_smt.passes.lowerers import SMTLowerer
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.comb_semantics import comb_semantics
from xdsl_smt.semantics.memref_semantics import memref_semantics
from xdsl_smt.passes.lower_to_smt import (
    func_to_smt_patterns,
    transfer_to_smt_patterns,
)
from xdsl_smt.pdl_constraints.integer_arith_constraints import (
    integer_arith_native_rewrites,
    integer_arith_native_constraints,
    integer_arith_native_static_constraints,
)
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
)
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.semantics.accessor import IntAccessor
from xdsl_smt.passes.lowerers_loaders import load_int_semantics_with_context
from xdsl_smt.pdl_constraints.parametric_integer_constraints import (
    parametric_integer_arith_native_rewrites,
    parametric_integer_arith_native_constraints,
    parametric_integer_arith_native_static_constraints,
)
from xdsl.pattern_rewriter import PatternRewriter
from xdsl_smt.passes.pdl_to_smt_context import PDLToSMTRewriteContext
from xdsl_smt.passes.pdl_to_smt_rewrites import (
    PatternRewrite,
    PatternStaticallyTrueRewrite,
    RewriteRewrite,
    TypeRewrite,
    AttributeRewrite,
    OperandRewrite,
    OperationRewrite,
    ReplaceRewrite,
    ResultRewrite,
    GetOpRewrite,
    AttachOpRewrite,
    ApplyNativeRewriteRewrite,
    ApplyNativeConstraintRewrite,
    ComputationOpRewrite,
    IntTypeRewrite,
    IntAttributeRewrite,
    IntOperandRewrite,
    StaticallyUnmatchedConstraintError,
)


@dataclass
class GenericPDLToSMTLowerer(ABC):
    native_rewrites: dict[
        str,
        Callable[[ApplyNativeRewriteOp, PatternRewriter, PDLToSMTRewriteContext], None],
    ] = field(default_factory=dict)
    native_constraints: dict[
        str,
        Callable[
            [ApplyNativeConstraintOp, PatternRewriter, PDLToSMTRewriteContext],
            SSAValue,
        ],
    ] = field(default_factory=dict)
    native_static_constraints: dict[
        str, Callable[[ApplyNativeConstraintOp, PDLToSMTRewriteContext], bool]
    ] = field(default_factory=dict)
    refinement: RefinementSemantics = IntegerTypeRefinementSemantics()

    def mark_pdl_operations(self, op: PatternOp):
        """
        Add an unit attribute with name "is_created" to operations that will be created.
        Add an unit attribute with name "is_deleted" to matching operations that will be deleted.
        """
        rewrite_op = op.body.ops.last
        if not isinstance(rewrite_op, RewriteOp):
            raise Exception(
                "Expected a rewrite operation at the end of the pdl pattern"
            )
        for sub_op in rewrite_op.walk():
            if isinstance(sub_op, OperationOp):
                sub_op.attributes["is_created"] = UnitAttr()
            if isinstance(sub_op, ReplaceOp):
                deleted_op = sub_op.op_value.owner
                assert isinstance(deleted_op, OperationOp)
                deleted_op.attributes["is_deleted"] = UnitAttr()

        # Check that operations in the matching part are first composed of non-deleted operations,
        # then deleted operations.
        # This is just to simplify the logic of the pdl to smt conversion, otherwise we would have
        # to treat effects differently.
        has_deleted = False
        for sub_op in op.body.ops:
            if not isinstance(sub_op, OperationOp):
                continue
            if "is_deleted" in sub_op.attributes:
                has_deleted = True
                continue
            if has_deleted:
                raise Exception(
                    "Operations in the matching part of the pdl pattern should be first"
                    "composed of non-deleted operations, then deleted operations"
                )

    @abstractmethod
    def select_integer_semantics(
        self, rewrite_context: PDLToSMTRewriteContext
    ) -> Tuple[OperandRewrite, AttributeRewrite, TypeRewrite]:
        pass

    def lower_to_smt(self, op: Operation, ctx: MLContext) -> None:
        patterns = [sub_op for sub_op in op.walk() if isinstance(sub_op, PatternOp)]
        n_patterns = len(patterns)
        if n_patterns > 1:
            raise Exception(
                f"Can only handle modules with a single pattern, found {n_patterns}"
            )
        if n_patterns == 0:
            return
        pattern = patterns[0]

        # First, mark all `pdl.operation` operations that will be created and deleted.
        # This helps other rewrite patterns to know which effects an operation should modify.
        self.mark_pdl_operations(pattern)

        # Set the input effect states
        insert_point = InsertPoint.at_start(pattern.body.blocks[0])
        new_state_op = DeclareConstOp(StateType())
        Rewriter.insert_op(new_state_op, insert_point)
        rewrite_context = PDLToSMTRewriteContext(new_state_op.res, new_state_op.res)

        (
            operand_rewrite,
            attribute_rewrite,
            type_rewrite,
        ) = self.select_integer_semantics(rewrite_context)

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RewriteRewrite(),
                    type_rewrite,
                    attribute_rewrite,
                    operand_rewrite,
                    GetOpRewrite(rewrite_context),
                    OperationRewrite(ctx, rewrite_context),
                    ReplaceRewrite(rewrite_context, self.refinement),
                    ResultRewrite(rewrite_context),
                    AttachOpRewrite(rewrite_context),
                    ApplyNativeRewriteRewrite(rewrite_context, self.native_rewrites),
                    ApplyNativeConstraintRewrite(
                        rewrite_context,
                        self.native_constraints,
                        self.native_static_constraints,
                    ),
                    ComputationOpRewrite(),
                ]
            )
        )
        try:
            walker.rewrite_op(op)
        except StaticallyUnmatchedConstraintError:
            PatternRewriteWalker(PatternStaticallyTrueRewrite()).rewrite_op(op)
        else:
            PatternRewriteWalker(PatternRewrite()).rewrite_op(op)


class IntPDLToSMTLowerer(GenericPDLToSMTLowerer):
    def select_integer_semantics(
        self, rewrite_context: PDLToSMTRewriteContext
    ) -> Tuple[OperandRewrite, AttributeRewrite, TypeRewrite]:
        if smt.CallOp in SMTLowerer.rewrite_patterns:
            del SMTLowerer.rewrite_patterns[smt.CallOp]
        accessor = IntAccessor()
        load_int_semantics_with_context(accessor, rewrite_context)
        self.refinement = IntIntegerTypeRefinementSemantics(accessor)
        self.native_rewrites = {
            **self.native_rewrites,
            **parametric_integer_arith_native_rewrites,
        }
        self.native_constraints = {
            **self.native_constraints,
            **parametric_integer_arith_native_constraints,
        }
        self.native_static_constraints = (
            parametric_integer_arith_native_static_constraints
        )
        operand_rewrite = IntOperandRewrite(rewrite_context, accessor)
        attribute_rewrite = IntAttributeRewrite()
        type_rewrite = IntTypeRewrite(rewrite_context)
        return (operand_rewrite, attribute_rewrite, type_rewrite)


class BVPDLToSMTLowerer(GenericPDLToSMTLowerer):
    def select_integer_semantics(
        self, rewrite_context: PDLToSMTRewriteContext
    ) -> Tuple[OperandRewrite, AttributeRewrite, TypeRewrite]:
        types = {
            IntegerType: IntegerTypeSemantics(),
            IndexType: IndexTypeSemantics(),
        }
        SMTLowerer.type_lowerers = {
            **SMTLowerer.type_lowerers,
            **types,
        }
        SMTLowerer.attribute_semantics = {IntegerAttr: IntegerAttrSemantics()}
        SMTLowerer.op_semantics = {
            **arith_semantics,
            **comb_semantics,
            **memref_semantics,
        }
        SMTLowerer.rewrite_patterns = {
            **func_to_smt_patterns,
            **transfer_to_smt_patterns,
        }
        self.native_rewrites = integer_arith_native_rewrites
        self.native_constraints = integer_arith_native_constraints
        self.native_static_constraints = integer_arith_native_static_constraints
        operand_rewrite = OperandRewrite()
        attribute_rewrite = AttributeRewrite()
        type_rewrite = TypeRewrite(rewrite_context)
        return (operand_rewrite, attribute_rewrite, type_rewrite)
