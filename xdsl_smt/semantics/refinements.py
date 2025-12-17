from typing import Sequence
from xdsl.dialects.builtin import FunctionType, IntegerType, i1
from xdsl.ir import SSAValue, Region, Block, Attribute
from xdsl.rewriter import InsertPoint
from xdsl.builder import Builder, ImplicitBuilder

from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType, UltOp
from xdsl_smt.dialects import smt_dialect as smt, smt_bitvector_dialect as smt_bv

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
from xdsl_smt.semantics.semantics import RefinementSemantics


class EqualityRefinementSemantics(RefinementSemantics):
    def get_semantics(
        self,
        val_before: SSAValue,
        val_after: SSAValue,
        builder: Builder,
    ) -> SSAValue:
        """Compute the refinement as equality between before and after values."""
        return builder.insert(smt.EqOp(val_before, val_after)).res


class IntegerTypeRefinementSemantics(RefinementSemantics):
    def get_semantics(
        self,
        val_before: SSAValue,
        val_after: SSAValue,
        builder: Builder,
    ) -> SSAValue:
        """Compute the refinement from a value with poison semantics to a value with poison semantics."""
        before_poison = builder.insert(SecondOp(val_before)).res
        after_poison = builder.insert(SecondOp(val_after)).res

        before_val = builder.insert(FirstOp(val_before)).res
        after_val = builder.insert(FirstOp(val_after)).res

        not_before_poison = builder.insert(smt.NotOp(before_poison)).result
        not_after_poison = builder.insert(smt.NotOp(after_poison)).result
        eq_vals = builder.insert(smt.EqOp(before_val, after_val)).res
        not_poison_eq = builder.insert(smt.AndOp(eq_vals, not_after_poison)).result
        refinement_integer = builder.insert(
            smt.ImpliesOp(not_before_poison, not_poison_eq)
        ).result
        return refinement_integer


class IntToIntPoisonRefinementSemantics(RefinementSemantics):
    def get_semantics(
        self,
        val_before: SSAValue,
        val_after: SSAValue,
        builder: Builder,
    ) -> SSAValue:
        """
        Compute the refinement from an integer without poison semantics
        to a integer with poison semantics.
        """
        before_val = val_before
        after_val = builder.insert(FirstOp(val_after)).res

        after_poison = builder.insert(SecondOp(val_after)).res
        not_after_poison = builder.insert(smt.NotOp(after_poison)).result

        eq_vals = builder.insert(smt.EqOp(before_val, after_val)).res
        refinement = builder.insert(smt.AndOp(eq_vals, not_after_poison)).result
        return refinement


class BoolToIntPoisonRefinementSemantics(RefinementSemantics):
    def get_semantics(
        self,
        val_before: SSAValue,
        val_after: SSAValue,
        builder: Builder,
    ) -> SSAValue:
        """
        Compute the refinement from an integer without poison semantics
        to a integer with poison semantics.
        """
        before_val = val_before
        after_val_int = builder.insert(FirstOp(val_after)).res
        zero = builder.insert(smt_bv.ConstantOp(0, 1)).res
        after_val = builder.insert(smt.DistinctOp(after_val_int, zero)).res

        after_poison = builder.insert(SecondOp(val_after)).res
        not_after_poison = builder.insert(smt.NotOp(after_poison)).result

        eq_vals = builder.insert(smt.EqOp(before_val, after_val)).res
        refinement = builder.insert(smt.AndOp(eq_vals, not_after_poison)).result
        return refinement


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


def find_refinement_semantics(
    type_before: Attribute,
    type_after: Attribute,
) -> RefinementSemantics:
    if isinstance(type_before, smt_bv.BitVectorType | smt.BoolType) and isinstance(
        type_after, smt_bv.BitVectorType | smt.BoolType
    ):
        return EqualityRefinementSemantics()
    if isinstance(type_before, IntegerType) and isinstance(type_after, IntegerType):
        return IntegerTypeRefinementSemantics()
    if isinstance(type_before, BitVectorType) and isinstance(type_after, IntegerType):
        return IntToIntPoisonRefinementSemantics()
    if isinstance(type_before, BoolType) and type_after == i1:
        return BoolToIntPoisonRefinementSemantics()
    raise Exception(f"No refinement semantics for types {type_before}, {type_after}")


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
        forall = ForallOp(Region(Block(arg_types=[BitVectorType(64)])))

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
        block_refinement = AndOp(block_properties_refinement, forall.result).result

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


def function_results_refinement(
    call_before: CallOp,
    function_type_before: FunctionType,
    call_after: CallOp,
    function_type_after: FunctionType,
    insert_point: InsertPoint,
) -> SSAValue[BoolType]:
    """
    Create operations to check that the results of a function call refines another.
    """
    builder = Builder(insert_point)
    # Refinement of non-state return values
    return_values_refinement = builder.insert(ConstantBoolOp(True)).result

    # Refines each non-state return value
    for ret, ret_after, original_type, original_type_after in zip(
        call_before.res[:-1],
        call_after.res[:-1],
        function_type_before.outputs.data,
        function_type_after.outputs.data,
        strict=True,
    ):
        refinement_semantics = find_refinement_semantics(
            original_type, original_type_after
        )
        value_refinement = refinement_semantics.get_semantics(ret, ret_after, builder)
        return_values_refinement = builder.insert(
            AndOp(return_values_refinement, value_refinement)
        ).result
    return_values_refinement.name_hint = "return_values_refinement"

    has_memory = isa(call_before.res[-1].type, PairType)

    # Refinement of memory
    if has_memory:
        mem_refinement = memory_refinement(call_before, call_after, insert_point)
    else:
        mem_refinement = builder.insert(ConstantBoolOp(True)).result
    mem_refinement.name_hint = "memory_refinement"

    with ImplicitBuilder(builder):
        # Get ub results
        if has_memory:
            res_ub = SecondOp(call_before.res[-1]).res
            res_ub_after = SecondOp(call_after.res[-1]).res
        else:
            res_ub = call_before.res[-1]
            res_ub_after = call_after.res[-1]

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


def insert_function_refinement_with_declare_const(
    func_before: DefineFunOp,
    func_type_before: FunctionType,
    func_after: DefineFunOp,
    func_type_after: FunctionType,
    insert_point: InsertPoint,
) -> SSAValue[BoolType]:
    """
    Create operations to check that one function refines another.
    Arguments passed to the functions are created with declare-const ops.
    """
    builder = Builder(insert_point)
    args_before: list[SSAValue] = []
    args_after: list[SSAValue] = []
    for (arg_before, arg_after), (type_before, type_after) in zip(
        zip(func_before.arg_types, func_after.arg_types, strict=True),
        zip(
            func_type_before.inputs,
            func_type_after.inputs,
            strict=True,
        ),
    ):
        if arg_before == arg_after and type_before == type_after:
            const_op = builder.insert(DeclareConstOp(arg_before))
            args_before.append(const_op.res)
            args_after.append(const_op.res)
        else:
            const_before = builder.insert(DeclareConstOp(arg_before))
            const_after = builder.insert(DeclareConstOp(arg_after))
            args_before.append(const_before.res)
            args_after.append(const_after.res)
            # Add refinement between arguments
            refinement_semantics = find_refinement_semantics(type_before, type_after)
            arg_refinement = refinement_semantics.get_semantics(
                const_before.res, const_after.res, builder
            )
            builder.insert(smt.AssertOp(arg_refinement))

    if len(func_before.arg_types) != len(func_type_before.inputs):
        assert len(func_before.arg_types) == len(func_type_before.inputs) + 1
        assert len(func_after.arg_types) == len(func_type_after.inputs) + 1
        arg = builder.insert(DeclareConstOp(func_before.arg_types[-1]))
        args_before.append(arg.res)
        args_after.append(arg.res)

    call_before = builder.insert(CallOp(func_before.ret, args_before))
    call_after = builder.insert(CallOp(func_after.ret, args_after))

    return function_results_refinement(
        call_before,
        func_type_before,
        call_after,
        func_type_after,
        insert_point,
    )


def insert_function_refinement_with_forall(
    func_before: DefineFunOp,
    func_type_before: FunctionType,
    func_after: DefineFunOp,
    func_type_after: FunctionType,
    insert_point: InsertPoint,
) -> SSAValue[BoolType]:
    """
    Create operations to check that one function refines another.
    This check uses an outside forall quantifier.
    """
    builder = Builder(insert_point)
    outer_forall_result: SSAValue | None = None
    args_before: list[SSAValue] = []
    args_after: list[SSAValue] = []
    preconditions: list[SSAValue] = []
    for (arg_before, arg_after), (type_before, type_after) in zip(
        zip(func_before.arg_types, func_after.arg_types, strict=True),
        zip(
            func_type_before.inputs,
            func_type_after.inputs,
            strict=True,
        ),
    ):
        forall: ForallOp
        if arg_before == arg_after and type_before == type_after:
            forall = builder.insert(ForallOp(Region(Block(arg_types=[arg_before]))))
            builder.insertion_point = InsertPoint.at_end(forall.body.block)
            args_before.append(forall.body.block.args[0])
            args_after.append(forall.body.block.args[0])
        else:
            forall = builder.insert(
                ForallOp(Region(Block(arg_types=[arg_before, arg_after])))
            )
            builder.insertion_point = InsertPoint.at_end(forall.body.block)
            args_before.append(forall.body.block.args[0])
            args_after.append(forall.body.block.args[1])
            # Add refinement between arguments
            refinement_semantics = find_refinement_semantics(type_before, type_after)
            arg_refinement = refinement_semantics.get_semantics(
                args_before[-1], args_after[-1], builder
            )
            preconditions.append(arg_refinement)
        # If we are in a forall, yield the value
        if outer_forall_result is not None:
            builder.insert_op(YieldOp(forall.result), InsertPoint.after(forall))
        if outer_forall_result is None:
            outer_forall_result = forall.result

    if len(func_before.arg_types) != len(func_type_before.inputs):
        assert len(func_before.arg_types) == len(func_type_before.inputs) + 1
        assert len(func_after.arg_types) == len(func_type_after.inputs) + 1
        forall = builder.insert(
            ForallOp(Region(Block(arg_types=[func_before.arg_types[-1]])))
        )
        builder.insertion_point = InsertPoint.at_end(forall.body.block)
        args_before.append(forall.body.block.args[0])
        args_after.append(forall.body.block.args[0])
        if outer_forall_result is not None:
            builder.insert_op(YieldOp(forall.result), InsertPoint.after(forall))
        if outer_forall_result is None:
            outer_forall_result = forall.result

    preconditions_conjunction: SSAValue = builder.insert(ConstantBoolOp(True)).result
    for precondition in preconditions:
        preconditions_conjunction = builder.insert(
            AndOp(preconditions_conjunction, precondition)
        ).result

    call_before = builder.insert(CallOp(func_before.ret, args_before))
    call_after = builder.insert(CallOp(func_after.ret, args_after))

    results_refinement = function_results_refinement(
        call_before,
        func_type_before,
        call_after,
        func_type_after,
        builder.insertion_point,
    )

    refinement = builder.insert(
        ImpliesOp(preconditions_conjunction, results_refinement)
    ).result

    if outer_forall_result is not None:
        builder.insert(YieldOp(refinement))
        return outer_forall_result
    return refinement
