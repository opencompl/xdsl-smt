from dataclasses import dataclass, field
from typing import Callable, ClassVar, Sequence
from xdsl.dialects.builtin import ModuleOp, IntegerType, StringAttr, UnitAttr, ArrayAttr
from xdsl.dialects.pdl import (
    ApplyNativeConstraintOp,
    ApplyNativeRewriteOp,
    OperandOp,
    OperationOp,
    PatternOp,
    ReplaceOp,
    ResultOp,
    RewriteOp,
    TypeOp,
    AttributeOp,
)
from xdsl.utils.hints import isa
from xdsl.rewriter import InsertPoint, Rewriter

from xdsl_smt.dialects.effects import ub_effect
from xdsl_smt.dialects.effects.effect import StateType
from xdsl_smt.semantics.refinements import find_refinement_semantics
from xdsl_smt.semantics.semantics import RefinementSemantics

from xdsl_smt.dialects import pdl_dataflow as pdl_dataflow
from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_utils_dialect as smt_utils
from xdsl_smt.dialects import smt_dialect as smt

from xdsl.ir import Attribute, ErasedSSAValue, Operation, SSAValue
from xdsl.context import Context

from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.passes import ModulePass
from ..dialects.smt_dialect import (
    AndOp,
    AssertOp,
    BoolType,
    CheckSatOp,
    ConstantBoolOp,
    DeclareConstOp,
    EqOp,
    NotOp,
    OrOp,
)
from xdsl_smt.passes.lower_to_smt.smt_lowerer import SMTLowerer
from xdsl_smt.passes.lower_to_smt import LowerToSMTPass


class StaticallyUnmatchedConstraintError(Exception):
    """
    Exception raised when the constraint is known to be statically unmatched.
    This is used for avoiding generating unverifying SMT operations. For example,
    if we know that one type should have a lower bitwidth than another, we can
    should statically check it, otherwise the SMT program we output will not verify
    for operations such as `extui`.
    """

    pass


@dataclass
class PDLToSMTRewriteContext:
    matching_effect_state: SSAValue
    rewriting_effect_state: SSAValue
    pdl_types_to_types: dict[SSAValue, Attribute] = field(
        default_factory=dict[SSAValue, Attribute]
    )
    pdl_op_to_values: dict[SSAValue, Sequence[SSAValue]] = field(
        default_factory=dict[SSAValue, Sequence[SSAValue]]
    )
    preconditions: list[SSAValue] = field(default_factory=list[SSAValue])


def _get_type_of_erased_type_value(value: SSAValue) -> Attribute:
    assert isinstance(value, ErasedSSAValue), "Error in rewriting logic"
    assert isinstance(
        (type_op := value.old_value.owner), TypeOp
    ), "Error in rewriting logic"
    type = type_op.constantType
    if type is None:
        raise Exception("Cannot handle non-constant types")
    return type


class PatternRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PatternOp, rewriter: PatternRewriter):
        rewriter.inline_block_before_matched_op(op.body.blocks[0])
        rewriter.insert_op_before_matched_op(CheckSatOp())
        rewriter.erase_matched_op()


class PatternStaticallyTrueRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PatternOp, rewriter: PatternRewriter):
        false_const = ConstantBoolOp(False)
        assert_false = AssertOp(false_const.result)
        check_sat = CheckSatOp()
        rewriter.replace_matched_op([false_const, assert_false, check_sat])


class RewriteRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RewriteOp, rewriter: PatternRewriter):
        if op.body is not None:
            rewriter.inline_block_before_matched_op(op.body.blocks[0])
        rewriter.erase_matched_op()


@dataclass
class TypeRewrite(RewritePattern):
    rewrite_context: PDLToSMTRewriteContext

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TypeOp, rewriter: PatternRewriter):
        if op.constantType is None:
            raise Exception("Cannot handle non-constant types")

        self.rewrite_context.pdl_types_to_types[op.result] = SMTLowerer.lower_type(
            op.constantType
        )
        rewriter.erase_matched_op(safe_erase=False)


@dataclass
class AttributeRewrite(RewritePattern):
    ctx: Context

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AttributeOp, rewriter: PatternRewriter):
        if op.value is not None:
            value = SMTLowerer.attribute_semantics[type(op.value)].get_semantics(
                op.value, rewriter
            )
            rewriter.replace_matched_op([], [value])
            return

        if op.value_type is not None:
            value_type = _get_type_of_erased_type_value(op.value_type)
            if not isinstance(value_type, IntegerType):
                raise Exception(
                    "Cannot handle quantification of attributes with non-integer types"
                )
            declare_op = DeclareConstOp(smt_bv.BitVectorType(value_type.width.data))
            rewriter.replace_matched_op(declare_op)
            return

        if (base_type_name := op.attributes.get("base_type")) is not None:
            if not isinstance(base_type_name, StringAttr):
                raise Exception("pdl base types should be string attributes")
            attr_def = self.ctx.get_optional_attr(base_type_name.data)
            if attr_def is None:
                raise Exception(
                    f"Cannot handle attributes of base type {base_type_name.data}, "
                    "it is not defined in the context"
                )
            value = SMTLowerer.attribute_semantics[attr_def].get_unbounded_semantics(
                rewriter
            )
            rewriter.replace_matched_op([], [value])
            return

        raise Exception("Cannot handle unbounded and untyped attributes")


class OperandRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: OperandOp, rewriter: PatternRewriter):
        if op.value_type is None:
            raise Exception("Cannot handle non-typed operands")
        type = _get_type_of_erased_type_value(op.value_type)
        smt_type = SMTLowerer.lower_type(type)
        rewriter.replace_matched_op(DeclareConstOp(smt_type))


@dataclass
class OperationRewrite(RewritePattern):
    ctx: Context
    rewrite_context: PDLToSMTRewriteContext

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: OperationOp, rewriter: PatternRewriter):
        # Get the corresponding op definition from the context
        if op.opName is None:
            raise Exception("Cannot handle non-constant op names")
        op_def = self.ctx.get_op(op.opName.data)

        # Create the with the given operands and types
        result_types = [_get_type_of_erased_type_value(type) for type in op.type_values]

        if op_def in SMTLowerer.op_semantics:
            attributes = {
                name.data: attr
                for name, attr in zip(op.attributeValueNames, op.attribute_values)
            }

            # If we are manipulating a created operation, we use the effect states of the rewriting part.
            # Otherwise, either we are manipulating an operation that is only in the matching part, or we are
            # manipulating an operation that is in both the matching and rewriting part, in which case both
            # effect states are the same.
            is_created = "is_created" in op.attributes
            is_deleted = "is_deleted" in op.attributes
            if is_created:
                effect_state = self.rewrite_context.rewriting_effect_state
            else:
                effect_state = self.rewrite_context.matching_effect_state
            results, new_effect_state = SMTLowerer.op_semantics[op_def].get_semantics(
                op.operand_values,
                result_types,
                attributes,
                effect_state,
                rewriter,
            )
            if new_effect_state is None:
                raise Exception(
                    "Cannot handle operations that do not return an effect state"
                )

            # Update the correct effect states.
            if is_deleted:
                self.rewrite_context.matching_effect_state = new_effect_state
            elif is_created:
                self.rewrite_context.rewriting_effect_state = new_effect_state
            else:
                self.rewrite_context.matching_effect_state = new_effect_state
                self.rewrite_context.rewriting_effect_state = new_effect_state

            self.rewrite_context.pdl_op_to_values[op.op] = results
            rewriter.erase_matched_op(safe_erase=False)
            return

        if op.attribute_values:
            raise Exception(
                f"operation {op.opName} is used with attributes, "
                "but no semantics are defined for this operation"
            )

        synthesized_op = op_def.create(
            operands=op.operand_values, result_types=result_types
        )

        # Cursed hack: we create a new module with that operation, and
        # we rewrite it with the arith_to_smt and comb_to_smt pass.
        rewrite_module = ModuleOp([synthesized_op])
        LowerToSMTPass().apply(self.ctx, rewrite_module)
        last_op = rewrite_module.body.block.last_op
        assert last_op is not None

        # Set the operation carrying the results in the context
        # FIXME: this does not work if the last operation does not return all results
        self.rewrite_context.pdl_op_to_values[op.op] = last_op.results
        rewriter.inline_block_before_matched_op(rewrite_module.body.blocks[0])

        rewriter.erase_matched_op(safe_erase=False)


@dataclass
class ReplaceRewrite(RewritePattern):
    rewrite_context: PDLToSMTRewriteContext
    refinement: RefinementSemantics | None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReplaceOp, rewriter: PatternRewriter):
        assert isinstance(op.op_value, ErasedSSAValue)
        replaced_values = self.rewrite_context.pdl_op_to_values[op.op_value.old_value]
        if len(replaced_values) != 1:
            raise Exception("Cannot handle operations with multiple results")
        replaced_value = replaced_values[0]

        replacing_value: SSAValue
        # Replacing by values case
        if len(op.repl_values) != 0:
            if len(op.repl_values) != 1:
                raise Exception("Cannot handle operations with multiple results")
            replacing_value = op.repl_values[0]
        # Replacing by operations case
        else:
            assert isinstance(op.repl_operation, ErasedSSAValue)
            replacing_values = self.rewrite_context.pdl_op_to_values[
                op.repl_operation.old_value
            ]
            if len(replacing_values) != 1:
                raise Exception("Cannot handle operations with multiple results")
            replacing_value = replacing_values[0]

        if self.refinement is None:
            replaced_types = op.attributes["__replaced_types"]
            replacing_types = op.attributes["__replacing_types"]
            assert isa(replaced_types, ArrayAttr[Attribute])
            assert isa(replacing_types, ArrayAttr[Attribute])
            refinement = find_refinement_semantics(
                replaced_types.data[0], replacing_types.data[0]
            )
        else:
            refinement = self.refinement

        value_refinement = refinement.get_semantics(
            replaced_value,
            replacing_value,
            rewriter,
        )
        # With UB, our refinement is: ub_before \/ (not ub_after /\ integer_refinement)
        ub_before_bool = ub_effect.ToBoolOp(self.rewrite_context.matching_effect_state)
        ub_after_bool = ub_effect.ToBoolOp(self.rewrite_context.rewriting_effect_state)
        not_ub_after = smt.NotOp(ub_after_bool.res)
        not_ub_before_case = smt.AndOp(not_ub_after.result, value_refinement)
        refinement = smt.OrOp(ub_before_bool.res, not_ub_before_case.result)
        rewriter.insert_op_before_matched_op(
            [
                ub_before_bool,
                ub_after_bool,
                not_ub_after,
                not_ub_before_case,
                refinement,
            ]
        )
        not_refinement = NotOp(refinement.result)
        rewriter.insert_op_before_matched_op(not_refinement)
        not_refinement_value = not_refinement.result

        if len(self.rewrite_context.preconditions) == 0:
            assert_op = AssertOp(not_refinement_value)
            rewriter.replace_matched_op([assert_op])
            return

        and_preconditions = self.rewrite_context.preconditions[0]
        for precondition in self.rewrite_context.preconditions[1:]:
            and_preconditions_op = AndOp(and_preconditions, precondition)
            rewriter.insert_op_before_matched_op(and_preconditions_op)
            and_preconditions = and_preconditions_op.result

        replace_correct = AndOp(not_refinement_value, and_preconditions)
        assert_op = AssertOp(replace_correct.result)
        rewriter.replace_matched_op([replace_correct, assert_op])


@dataclass
class ResultRewrite(RewritePattern):
    rewrite_context: PDLToSMTRewriteContext

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ResultOp, rewriter: PatternRewriter):
        assert isinstance(op.parent_, ErasedSSAValue)
        result = self.rewrite_context.pdl_op_to_values[op.parent_.old_value][
            op.index.value.data
        ]
        rewriter.replace_matched_op([], new_results=[result])


def kb_analysis_correct(
    poisoned_value: SSAValue, zeros: SSAValue, ones: SSAValue
) -> tuple[SSAValue, list[Operation]]:
    assert isa(poisoned_value.type, smt_utils.PairType[smt_bv.BitVectorType, BoolType])

    value_op = smt_utils.FirstOp(poisoned_value)
    value = value_op.res
    assert isinstance(value.type, smt_bv.BitVectorType)

    poison_op = smt_utils.SecondOp(poisoned_value)
    poison = poison_op.res
    true_op = ConstantBoolOp(True)
    poison_correct = EqOp(poison, true_op.result)

    and_op_zeros = smt_bv.AndOp(value, zeros)
    zero = smt_bv.ConstantOp(0, value.type.width.data)
    zeros_correct = EqOp(and_op_zeros.res, zero.res)
    and_op_ones = smt_bv.AndOp(value, ones)
    ones_correct = EqOp(and_op_ones.res, ones)
    value_correct = AndOp(zeros_correct.res, ones_correct.res)
    correct = OrOp(poison_correct.res, value_correct.result)

    return value_correct.result, [
        value_op,
        poison_op,
        true_op,
        poison_correct,
        and_op_zeros,
        zero,
        zeros_correct,
        and_op_ones,
        ones_correct,
        value_correct,
        correct,
    ]


@dataclass
class GetOpRewrite(RewritePattern):
    rewrite_context: PDLToSMTRewriteContext

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: pdl_dataflow.GetOp, rewriter: PatternRewriter):
        assert isa(op.value.type, smt_utils.PairType[smt_bv.BitVectorType, BoolType])
        bv_type = op.value.type.first
        zeros_op = DeclareConstOp(bv_type)
        ones_op = DeclareConstOp(bv_type)

        value = op.value
        zeros = zeros_op.res
        ones = ones_op.res

        all_correct, correct_ops = kb_analysis_correct(value, zeros, ones)
        self.rewrite_context.preconditions.append(all_correct)

        rewriter.replace_matched_op(
            [
                zeros_op,
                ones_op,
                *correct_ops,
            ],
            new_results=[zeros, ones],
        )
        name = op.value.name_hint if op.value.name_hint else "value"
        zeros.name_hint = name + "_zeros"
        ones.name_hint = name + "_ones"


@dataclass
class AttachOpRewrite(RewritePattern):
    rewrite_context: PDLToSMTRewriteContext

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: pdl_dataflow.AttachOp, rewriter: PatternRewriter):
        assert len(op.domains) == 2
        zeros, ones = op.domains

        analysis_correct, analysis_correct_ops = kb_analysis_correct(
            op.value, zeros, ones
        )
        rewriter.insert_op_before_matched_op(analysis_correct_ops)
        analysis_incorrect_op = NotOp(analysis_correct)
        rewriter.insert_op_before_matched_op(analysis_incorrect_op)
        analysis_incorrect = analysis_incorrect_op.result

        if len(self.rewrite_context.preconditions) == 0:
            rewriter.replace_matched_op(AssertOp(analysis_incorrect))
            return

        and_preconditions = self.rewrite_context.preconditions[0]
        for precondition in self.rewrite_context.preconditions[1:]:
            and_preconditions_op = AndOp(and_preconditions, precondition)
            rewriter.insert_op_before_matched_op(and_preconditions_op)
            and_preconditions = and_preconditions_op.result

        implies = AndOp(and_preconditions, analysis_incorrect)
        rewriter.insert_op_before_matched_op(implies)
        rewriter.replace_matched_op(AssertOp(implies.result))


@dataclass
class ApplyNativeRewriteRewrite(RewritePattern):
    rewrite_context: PDLToSMTRewriteContext
    native_rewrites: dict[
        str,
        Callable[[ApplyNativeRewriteOp, PatternRewriter, PDLToSMTRewriteContext], None],
    ] = field(
        default_factory=dict[
            str,
            Callable[
                [ApplyNativeRewriteOp, PatternRewriter, PDLToSMTRewriteContext], None
            ],
        ]
    )

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyNativeRewriteOp, rewriter: PatternRewriter, /):
        if op.constraint_name.data not in self.native_rewrites:
            raise Exception(
                f"No semantics for native rewrite {op.constraint_name.data}"
            )
        rewrite = self.native_rewrites[op.constraint_name.data]
        rewrite(op, rewriter, self.rewrite_context)


@dataclass
class ApplyNativeConstraintRewrite(RewritePattern):
    rewrite_context: PDLToSMTRewriteContext
    native_constraints: dict[
        str,
        Callable[
            [ApplyNativeConstraintOp, PatternRewriter, PDLToSMTRewriteContext], SSAValue
        ],
    ] = field(
        default_factory=dict[
            str,
            Callable[
                [ApplyNativeConstraintOp, PatternRewriter, PDLToSMTRewriteContext],
                SSAValue,
            ],
        ]
    )
    native_static_constraints: dict[
        str, Callable[[ApplyNativeConstraintOp, PDLToSMTRewriteContext], bool]
    ] = field(
        default_factory=dict[
            str, Callable[[ApplyNativeConstraintOp, PDLToSMTRewriteContext], bool]
        ]
    )

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: ApplyNativeConstraintOp, rewriter: PatternRewriter, /
    ):
        if op.constraint_name.data in self.native_constraints:
            constraint = self.native_constraints[op.constraint_name.data]
            value = constraint(op, rewriter, self.rewrite_context)
            self.rewrite_context.preconditions.append(value)
            return
        if op.constraint_name.data in self.native_static_constraints:
            constraint = self.native_static_constraints[op.constraint_name.data]
            if not constraint(op, self.rewrite_context):
                raise StaticallyUnmatchedConstraintError()
            rewriter.erase_matched_op()
            return
        raise Exception(f"No semantics for native constraint {op.constraint_name.data}")


class ComputationOpRewrite(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if (
            type(op) in SMTLowerer.op_semantics
            or type(op) in SMTLowerer.rewrite_patterns
        ):
            new_effects = SMTLowerer.lower_operation(op, None)
            assert (
                new_effects is None
            ), "Operations used as computations in PDL should not have effects"


def annotate_replace_op(pattern: PatternOp):
    """
    Annotate all `pdl.replace` operations in the given pattern with the types of the
    replaced and replacing values.
    """
    values_to_types: dict[SSAValue, tuple[Attribute, ...]] = {}
    for op in pattern.walk():
        if isinstance(op, OperandOp):
            if op.value_type is None:
                raise Exception("Cannot handle non-typed operands")
            owner = op.value_type.owner
            assert isinstance(owner, TypeOp)
            assert owner.constantType is not None
            values_to_types[op.value] = (owner.constantType,)
        if isinstance(op, OperationOp):
            values: list[Attribute] = []
            for type_value in op.type_values:
                owner = type_value.owner
                assert isinstance(owner, TypeOp)
                assert owner.constantType is not None
                values.append(owner.constantType)
            values_to_types[op.op] = tuple(values)
        if isinstance(op, ResultOp):
            values_to_types[op.val] = (
                values_to_types[op.parent_][op.index.value.data],
            )
        if isinstance(op, ReplaceOp):
            replaced_value_types = values_to_types[op.op_value]
            if len(op.repl_values) != 0:
                replacing_value_types: list[Attribute] = []
                for repl_value in op.repl_values:
                    replacing_value_types.append(values_to_types[repl_value][0])
                replacing_types = tuple(replacing_value_types)
            else:
                assert op.repl_operation is not None
                replacing_types = values_to_types[op.repl_operation]
            op.attributes["__replaced_types"] = ArrayAttr(replaced_value_types)
            op.attributes["__replacing_types"] = ArrayAttr(replacing_types)


@dataclass
class PDLToSMTLowerer:
    native_rewrites: dict[
        str,
        Callable[[ApplyNativeRewriteOp, PatternRewriter, PDLToSMTRewriteContext], None],
    ] = field(
        default_factory=dict[
            str,
            Callable[
                [ApplyNativeRewriteOp, PatternRewriter, PDLToSMTRewriteContext], None
            ],
        ]
    )
    native_constraints: dict[
        str,
        Callable[
            [ApplyNativeConstraintOp, PatternRewriter, PDLToSMTRewriteContext],
            SSAValue,
        ],
    ] = field(
        default_factory=dict[
            str,
            Callable[
                [ApplyNativeConstraintOp, PatternRewriter, PDLToSMTRewriteContext],
                SSAValue,
            ],
        ]
    )
    native_static_constraints: dict[
        str, Callable[[ApplyNativeConstraintOp, PDLToSMTRewriteContext], bool]
    ] = field(
        default_factory=dict[
            str, Callable[[ApplyNativeConstraintOp, PDLToSMTRewriteContext], bool]
        ]
    )
    refinement: RefinementSemantics | None = None

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

    def lower_to_smt(self, module: ModuleOp, ctx: Context) -> None:
        patterns = [sub_op for sub_op in module.walk() if isinstance(sub_op, PatternOp)]
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
        annotate_replace_op(pattern)

        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RewriteRewrite(),
                    TypeRewrite(rewrite_context),
                    AttributeRewrite(ctx),
                    OperandRewrite(),
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
            walker.rewrite_module(module)
        except StaticallyUnmatchedConstraintError:
            PatternRewriteWalker(PatternStaticallyTrueRewrite()).rewrite_module(module)
        else:
            PatternRewriteWalker(PatternRewrite()).rewrite_module(module)


@dataclass(frozen=True)
class PDLToSMT(ModulePass):
    name = "pdl-to-smt"

    pdl_lowerer: ClassVar[PDLToSMTLowerer] = PDLToSMTLowerer({}, {}, {})

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        self.pdl_lowerer.lower_to_smt(op, ctx)
