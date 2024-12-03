from dataclasses import dataclass, field
from typing import Callable
from xdsl.dialects.builtin import ModuleOp, IntegerType
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
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import transfer
from xdsl_smt.semantics.semantics import RefinementSemantics

from ..dialects import pdl_dataflow as pdl_dataflow
from ..dialects import smt_bitvector_dialect as smt_bv
from ..dialects import smt_utils_dialect as smt_utils

from xdsl.pattern_rewriter import PatternRewriter
from xdsl.ir import Attribute, ErasedSSAValue, Operation, SSAValue
from xdsl.context import MLContext

from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
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
from xdsl_smt.passes.lower_to_smt import LowerToSMTPass
from xdsl_smt.passes.lowerers import SMTLowerer
from xdsl_smt.semantics.accessor import IntAccessor
from xdsl_smt.passes.pdl_to_smt_context import PDLToSMTRewriteContext


class StaticallyUnmatchedConstraintError(Exception):
    """
    Exception raised when the constraint is known to be statically unmatched.
    This is used for avoiding generating unverifying SMT operations. For example,
    if we know that one type should have a lower bitwidth than another, we can
    should statically check it, otherwise the SMT program we output will not verify
    for operations such as `extui`.
    """

    pass


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
        false_const = ConstantBoolOp.from_bool(False)
        assert_false = AssertOp(false_const.res)
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

    def _match_and_rewrite(self, op: TypeOp, rewriter: PatternRewriter):
        pass

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: TypeOp, rewriter: PatternRewriter):
        if op.constantType is None:
            raise Exception("Cannot handle non-constant types")

        self.rewrite_context.pdl_types_to_types[op.result] = SMTLowerer.lower_type(
            op.constantType
        )

        self._match_and_rewrite(op, rewriter)

        rewriter.erase_matched_op(safe_erase=False)


class AttributeRewrite(RewritePattern):
    rewrite_context: PDLToSMTRewriteContext

    def declare_of_value_type(self, value_type: Attribute):
        if not isinstance(value_type, IntegerType):
            raise Exception(
                "Cannot handle quantification of attributes with non-integer types"
            )
        declare_op = DeclareConstOp(smt_bv.BitVectorType(value_type.width.data))
        return declare_op

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
            declare_op = self.declare_of_value_type(value_type)
            rewriter.replace_matched_op(declare_op)
            return

        raise Exception("Cannot handle unbounded and untyped attributes")


class OperandRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: OperandOp, rewriter: PatternRewriter):
        if op.value_type is None:
            raise Exception("Cannot handle non-typed operands")
        type = _get_type_of_erased_type_value(op.value_type)
        smt_type = SMTLowerer.lower_type(type)
        declare_const_op = DeclareConstOp(smt_type)
        rewriter.replace_matched_op(declare_const_op)


@dataclass
class OperationRewrite(RewritePattern):
    ctx: MLContext
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

            # If we are manipulating a created operation, we use the
            # effect states of the rewriting part. Otherwise, either
            # we are manipulating an operation that is only in the
            # matching part, or we are manipulating an operation that
            # is in both the matching and rewriting part, in which
            # case both effect states are the same.
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
    refinement: RefinementSemantics

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

        refinement_value = self.refinement.get_semantics(
            replaced_value,
            replacing_value,
            self.rewrite_context.matching_effect_state,
            self.rewrite_context.rewriting_effect_state,
            rewriter,
        )
        not_refinement = NotOp(refinement_value)
        rewriter.insert_op_before_matched_op(not_refinement)
        not_refinement_value = not_refinement.res

        if len(self.rewrite_context.preconditions) == 0:
            assert_op = AssertOp(not_refinement_value)
            rewriter.replace_matched_op([assert_op])
            return

        and_preconditions = self.rewrite_context.preconditions[0]
        for precondition in self.rewrite_context.preconditions[1:]:
            and_preconditions_op = AndOp(and_preconditions, precondition)
            rewriter.insert_op_before_matched_op(and_preconditions_op)
            and_preconditions = and_preconditions_op.res
        replace_correct = AndOp(not_refinement_value, and_preconditions)
        assert_op = AssertOp(replace_correct.res)
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
    true_op = ConstantBoolOp.from_bool(True)
    poison_correct = EqOp(poison, true_op.res)

    and_op_zeros = smt_bv.AndOp(value, zeros)
    zero = smt_bv.ConstantOp(0, value.type.width.data)
    zeros_correct = EqOp(and_op_zeros.res, zero.res)
    and_op_ones = smt_bv.AndOp(value, ones)
    ones_correct = EqOp(and_op_ones.res, ones)
    value_correct = AndOp(zeros_correct.res, ones_correct.res)
    correct = OrOp(poison_correct.res, value_correct.res)

    return value_correct.res, [
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
        analysis_incorrect_op = NotOp.get(analysis_correct)
        rewriter.insert_op_before_matched_op(analysis_incorrect_op)
        analysis_incorrect = analysis_incorrect_op.res

        if len(self.rewrite_context.preconditions) == 0:
            rewriter.replace_matched_op(AssertOp(analysis_incorrect))
            return

        and_preconditions = self.rewrite_context.preconditions[0]
        for precondition in self.rewrite_context.preconditions[1:]:
            and_preconditions_op = AndOp(and_preconditions, precondition)
            rewriter.insert_op_before_matched_op(and_preconditions_op)
            and_preconditions = and_preconditions_op.res

        implies = AndOp(and_preconditions, analysis_incorrect)
        rewriter.insert_op_before_matched_op(implies)
        rewriter.replace_matched_op(AssertOp(implies.res))


@dataclass
class ApplyNativeRewriteRewrite(RewritePattern):
    rewrite_context: PDLToSMTRewriteContext
    native_rewrites: dict[
        str,
        Callable[[ApplyNativeRewriteOp, PatternRewriter, PDLToSMTRewriteContext], None],
    ] = field(default_factory=dict)

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
    ] = field(default_factory=dict)
    native_static_constraints: dict[
        str, Callable[[ApplyNativeConstraintOp, PDLToSMTRewriteContext], bool]
    ] = field(default_factory=dict)

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


@dataclass
class IntTypeRewrite(TypeRewrite):
    rewrite_context: PDLToSMTRewriteContext

    def _match_and_rewrite(self, op: TypeOp, rewriter: PatternRewriter):
        if isinstance(op.constantType, IntegerType):
            width_op = smt_int.ConstantOp(op.constantType.width.data)
        elif isinstance(op.constantType, transfer.TransIntegerType):
            width_op = DeclareConstOp(smt_int.SMTIntType())
        else:
            assert False
        rewriter.insert_op_before_matched_op(width_op)
        self.rewrite_context.pdl_types_to_width[op.result] = width_op.res


class IntAttributeRewrite(AttributeRewrite):
    rewrite_context: PDLToSMTRewriteContext

    def declare_of_value_type(self, value_type: Attribute):
        if not isinstance(value_type, IntegerType) and not isinstance(
            value_type, transfer.TransIntegerType
        ):
            raise Exception(
                "Cannot handle quantification of attributes with non-integer types"
            )
        declare_op = DeclareConstOp(smt_int.SMTIntType())
        return declare_op


@dataclass
class IntOperandRewrite(OperandRewrite):
    rewrite_context: PDLToSMTRewriteContext
    accessor: IntAccessor

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: OperandOp, rewriter: PatternRewriter):
        if op.value_type is None:
            raise Exception("Cannot handle non-typed operands")
        type = _get_type_of_erased_type_value(op.value_type)
        # Payload
        payload_op = DeclareConstOp(smt_int.SMTIntType())
        # Poison
        poison_op = DeclareConstOp(smt.BoolType())
        # Width
        rewriter.insert_op_before_matched_op([payload_op, poison_op])
        assert isinstance(op.value_type, ErasedSSAValue)
        if op.value_type.old_value in self.rewrite_context.pdl_types_to_width:
            width = self.rewrite_context.pdl_types_to_width[op.value_type.old_value]
        else:
            if isinstance(type, IntegerType):
                width_op = smt_int.ConstantOp(type.width.data)
                rewriter.insert_op_before_matched_op(width_op)
                width = width_op.res
                self.rewrite_context.pdl_types_to_width[op.value_type] = width
            elif isinstance(type, transfer.TransIntegerType):
                width_op = DeclareConstOp(smt_int.SMTIntType())
                rewriter.insert_op_before_matched_op(width_op)
                width = width_op.res
                self.rewrite_context.pdl_types_to_width[op.value_type] = width
            else:
                assert False
        packed_integer = self.accessor.get_packed_integer(
            payload_op.res, poison_op.res, width, rewriter
        )

        assert isinstance(packed_integer.owner, Operation)
        packed_integer.owner.detach()
        rewriter.replace_matched_op(packed_integer.owner)
        if isinstance(type, IntegerType):
            return
        # Constraints on the operand
        rewriter = PatternRewriter(current_operation=packed_integer.owner)
        x = self.accessor.get_payload(packed_integer, rewriter)
        zero_op = smt_int.ConstantOp(0)
        rewriter.insert_op_after_matched_op([zero_op])
        # Needs to be greater or equal than zero
        x_ge_zero_op = smt_int.GeOp(x, zero_op.res)
        assert0_op = smt.AssertOp(x_ge_zero_op.res)
        rewriter.insert_op_after_matched_op([x_ge_zero_op, assert0_op])
        # Needs to be less or equal thant the maximum value
        int_max = self.accessor.get_int_max(
            packed_integer, smt_int.SMTIntType(), rewriter
        )
        x_le_int_max_op = smt_int.LtOp(x, int_max)
        assert1_op = smt.AssertOp(x_le_int_max_op.res)
        rewriter.insert_op_after_matched_op([x_le_int_max_op, assert1_op])
        # The width needs to be greater than zero
        width = self.accessor.get_width(packed_integer, rewriter)
        width_gt_zero_op = smt_int.GtOp(width, zero_op.res)
        assert2_op = smt.AssertOp(width_gt_zero_op.res)
        rewriter.insert_op_after_matched_op([width_gt_zero_op, assert2_op])
