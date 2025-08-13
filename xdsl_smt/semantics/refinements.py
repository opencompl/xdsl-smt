from typing import Sequence
from xdsl.dialects.builtin import FunctionType, IntegerType
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.builder import Builder, ImplicitBuilder

from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType, UltOp
import xdsl_smt.dialects.smt_dialect as smt

from xdsl.utils.hints import isa
from xdsl_smt.dialects import memory_dialect as mem
from xdsl_smt.dialects.memory_dialect import BlockIDType
from xdsl_smt.dialects.smt_dialect import (
    BoolType,
    ConstantBoolOp,
    DeclareConstOp,
    DefineFunOp,
    ForallOp,
    IteOp,
    NotOp,
    EqOp,
    AndOp,
    OrOp,
    ImpliesOp,
    YieldOp,
    CallOp,
)
from xdsl_smt.dialects.smt_utils_dialect import FirstOp, PairType, SecondOp, AnyPairType
from xdsl_smt.dialects.effects import ub_effect
import xdsl_smt.dialects.smt_utils_dialect as smt_utils
from xdsl_smt.semantics.semantics import RefinementSemantics


class IntegerTypeRefinementSemantics(RefinementSemantics):
    def get_semantics(
        self,
        val_before: SSAValue,
        val_after: SSAValue,
        state_before: SSAValue,
        state_after: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        """Compute the refinement from a value with poison semantics to a value with poison semantics."""
        before_poison = smt_utils.SecondOp(val_before)
        after_poison = smt_utils.SecondOp(val_after)

        before_val = smt_utils.FirstOp(val_before)
        after_val = smt_utils.FirstOp(val_after)

        rewriter.insert_op_before_matched_op(
            [
                before_poison,
                after_poison,
                before_val,
                after_val,
            ]
        )

        not_before_poison = smt.NotOp(before_poison.res)
        not_after_poison = smt.NotOp(after_poison.res)
        eq_vals = smt.EqOp(before_val.res, after_val.res)
        not_poison_eq = smt.AndOp(eq_vals.res, not_after_poison.result)
        refinement_integer = smt.ImpliesOp(
            not_before_poison.result, not_poison_eq.result
        )
        rewriter.insert_op_before_matched_op(
            [
                not_before_poison,
                not_after_poison,
                eq_vals,
                not_poison_eq,
                refinement_integer,
            ]
        )

        # With UB, our refinement is: ub_before \/ (not ub_after /\ integer_refinement)
        ub_before_bool = ub_effect.ToBoolOp(state_before)
        ub_after_bool = ub_effect.ToBoolOp(state_after)
        not_ub_after = smt.NotOp(ub_after_bool.res)
        not_ub_before_case = smt.AndOp(not_ub_after.result, refinement_integer.result)
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
        return refinement.result


def integer_value_refinement(
    value: SSAValue, value_after: SSAValue, insert_point: InsertPoint
) -> SSAValue:
    with ImplicitBuilder(Builder(insert_point)):
        not_after_poison = NotOp(SecondOp(value_after).res).result
        value_eq = EqOp.get(FirstOp(value).res, FirstOp(value_after).res).res
        value_refinement = AndOp(not_after_poison, value_eq).result
        refinement = OrOp(value_refinement, SecondOp(value).res).result
    return refinement


def get_block_ids_from_value(
    val: SSAValue, insert_point: InsertPoint
) -> list[SSAValue]:
    """Get all block ids that are used in the value."""
    if isinstance(val.type, BlockIDType):
        return [val]
    if isa(val.type, AnyPairType):
        with ImplicitBuilder(Builder(insert_point)):
            first = FirstOp(val).res
            second = SecondOp(val).res
        return get_block_ids_from_value(first, insert_point) + get_block_ids_from_value(
            second, insert_point
        )
    return []


def get_mapped_block_id(
    output_blockids: Sequence[SSAValue],
    output_blockids_after: Sequence[SSAValue],
    value: SSAValue,
    insert_point: InsertPoint,
):
    """
    Get the corresponding output block id from an input memory block id.
    """
    # Default case: The block id is one from the input, so it is
    # mapped to itself (input block id's do not change between before and after
    # in functions).
    result_value = value

    # Check for each output value its equality, and if equal, replace the
    # result value with the corresponding output value after.
    with ImplicitBuilder(Builder(insert_point)):
        for output, output_after in zip(output_blockids, output_blockids_after):
            is_eq = EqOp.get(value, output).res
            result_value = IteOp(is_eq, output_after, result_value).res

    return result_value


def memory_block_refinement(
    block: SSAValue,
    block_after: SSAValue,
    insert_point: InsertPoint,
) -> SSAValue:
    """
    Check refinement of two memory blocks.
    """

    with ImplicitBuilder(Builder(insert_point)):
        size = mem.GetBlockSizeOp(block).res
        size_after = mem.GetBlockSizeOp(block_after).res
        live = mem.GetBlockLiveMarkerOp(block).res
        live_after = mem.GetBlockLiveMarkerOp(block_after).res
        bytes = mem.GetBlockBytesOp(block).res
        bytes_after = mem.GetBlockBytesOp(block_after).res

        # Forall index, bytes[index] >= bytes_after[index]
        forall = ForallOp.from_variables([BitVectorType(64)])

        forall_block = forall.body.block

    with ImplicitBuilder(Builder(InsertPoint.at_end(forall_block))):
        in_bounds = UltOp(forall_block.args[0], size).res
        value = mem.ReadBytesOp(
            bytes, forall_block.args[0], PairType(BitVectorType(8), BoolType())
        ).res
        value_after = mem.ReadBytesOp(
            bytes_after, forall_block.args[0], PairType(BitVectorType(8), BoolType())
        ).res
    value_refinement = integer_value_refinement(
        value, value_after, InsertPoint.at_end(forall_block)
    )
    with ImplicitBuilder(Builder(InsertPoint.at_end(forall_block))):
        block_refinement = ImpliesOp(in_bounds, value_refinement).result
        YieldOp(block_refinement)

    with ImplicitBuilder(Builder(insert_point)):
        size_refinement = EqOp(size, size_after).res
        live_refinement = EqOp(live, live_after).res
        block_properties_refinement = AndOp(size_refinement, live_refinement).result
        block_refinement = AndOp(block_properties_refinement, forall.res).result

    return block_refinement


def memory_refinement(
    func_call: CallOp,
    func_call_after: CallOp,
    insert_point: InsertPoint,
) -> SSAValue:
    """Check refinement of two memory states."""

    # Get references to input and output block ids
    with ImplicitBuilder(Builder(insert_point)):
        memory = FirstOp(func_call.res[-1]).res
        memory_after = FirstOp(func_call_after.res[-1]).res

    input_block_ids = list[SSAValue]()
    ret_block_ids_before = list[SSAValue]()
    ret_block_ids_after = list[SSAValue]()

    for arg in func_call.args[:-1]:
        input_block_ids += get_block_ids_from_value(arg, insert_point)
    for ret, ret_after in zip(func_call.res[:-1], func_call_after.res[:-1]):
        ret_block_ids_before += get_block_ids_from_value(ret, insert_point)
        ret_block_ids_after += get_block_ids_from_value(ret_after, insert_point)

    accessible_block_ids = set(input_block_ids + ret_block_ids_before)

    with ImplicitBuilder(Builder(insert_point)):
        refinement = ConstantBoolOp(True).result

    for block_id in accessible_block_ids:
        block_id_after = get_mapped_block_id(
            ret_block_ids_before, ret_block_ids_after, block_id, insert_point
        )
        with ImplicitBuilder(Builder(insert_point)):
            block = mem.GetBlockOp(memory, block_id).res
            block_after = mem.GetBlockOp(memory_after, block_id_after).res

        block_refinement = memory_block_refinement(block, block_after, insert_point)
        with ImplicitBuilder(Builder(insert_point)):
            refinement = AndOp(refinement, block_refinement).result

    return refinement


def add_function_refinement(
    func: DefineFunOp,
    func_after: DefineFunOp,
    function_type: FunctionType,
    insert_point: InsertPoint,
    *,
    args: Sequence[SSAValue] | None = None,
) -> SSAValue:
    """
    Create operations to check that one function refines another.
    An assert check is added to the end of the list of operations.
    """
    builder = Builder(insert_point)
    with ImplicitBuilder(builder):
        # Quantify over all arguments
        if args is None:
            args = []
            for arg in func.body.blocks[0].args:
                const_op = DeclareConstOp(arg.type)
                args.append(const_op.res)

        # Call both operations
        func_call = CallOp(func.ret, args)
        func_call_after = CallOp(func_after.ret, args)

        # Refinement of non-state return values
        return_values_refinement = ConstantBoolOp(True).result

        # Refines each non-state return value
        for (ret, ret_after), original_type in zip(
            zip(func_call.res[:-1], func_call_after.res[:-1], strict=True),
            function_type.outputs.data,
            strict=True,
        ):
            if not isinstance(original_type, IntegerType):
                raise Exception("Cannot handle non-integer return types")
            not_after_poison = NotOp(SecondOp(ret_after).res)
            value_eq = EqOp.get(FirstOp(ret).res, FirstOp(ret_after).res)
            value_refinement = AndOp(not_after_poison.result, value_eq.res)
            refinement = OrOp(value_refinement.result, SecondOp(ret).res)
            return_values_refinement = AndOp(
                return_values_refinement, refinement.result
            ).result
        return_values_refinement.name_hint = "return_values_refinement"

    # Refinement of memory
    mem_refinement = memory_refinement(func_call, func_call_after, insert_point)
    mem_refinement.name_hint = "memory_refinement"

    with ImplicitBuilder(builder):
        # Get ub results
        res_ub = SecondOp(func_call.res[-1]).res
        res_ub_after = SecondOp(func_call_after.res[-1]).res

        res_ub.name_hint = "ub"
        res_ub_after.name_hint = "ub_after"

        # Compute refinement with UB
        refinement = OrOp(
            AndOp(
                NotOp(res_ub_after).result,
                AndOp(return_values_refinement, mem_refinement).result,
            ).result,
            res_ub,
        ).result
        refinement.name_hint = "function_refinement"

    return refinement
