#!/usr/bin/env python3

"""Check that a function refines another function."""

import argparse
import sys
from typing import Sequence

from xdsl.context import MLContext
from xdsl.ir import Operation, SSAValue
from xdsl.parser import Parser
from xdsl.rewriter import Rewriter, InsertPoint
from xdsl.builder import ImplicitBuilder, Builder
from xdsl.utils.isattr import isattr

from xdsl_smt.dialects.memory_dialect import BlockIDType
from xdsl_smt.dialects import memory_dialect as mem
from xdsl_smt.passes.dead_code_elimination import DeadCodeElimination
from xdsl_smt.passes.lower_effects_with_memory import (
    LowerEffectsWithMemoryPass,
)
from xdsl_smt.passes.lower_memory_effects import LowerMemoryEffectsPass
from xdsl_smt.passes.lower_memory_to_array import LowerMemoryToArrayPass

from xdsl_smt.dialects.smt_bitvector_dialect import (
    BitVectorType,
    SMTBitVectorDialect,
    UltOp,
)
from xdsl_smt.dialects.smt_dialect import (
    BoolType,
    AndOp,
    CallOp,
    CheckSatOp,
    ConstantBoolOp,
    DeclareConstOp,
    DefineFunOp,
    EqOp,
    AssertOp,
    ForallOp,
    ImpliesOp,
    IteOp,
    NotOp,
    OrOp,
    ReturnOp,
    SMTDialect,
    YieldOp,
)
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.effects.ub_effect import UBEffectDialect
from xdsl_smt.dialects.effects.effect import EffectDialect
from xdsl_smt.dialects.smt_utils_dialect import (
    FirstOp,
    SMTUtilsDialect,
    SecondOp,
    PairType,
    AnyPairType,
)
from xdsl_smt.dialects.hw_dialect import HW
from xdsl_smt.dialects.llvm_dialect import LLVM
from xdsl.dialects.builtin import (
    Builtin,
    ModuleOp,
    IntegerType,
    FunctionType,
)
from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.dialects.memref import MemRef

from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)
from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import load_vanilla_semantics
from xdsl_smt.passes.lower_to_smt import (
    LowerToSMTPass,
)
from xdsl_smt.passes.transfer_inline import FunctionCallInline
from xdsl_smt.traits.smt_printer import print_to_smtlib


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "before_file", type=str, nargs="?", help="path to before input file"
    )

    arg_parser.add_argument(
        "after_file", type=str, nargs="?", help="path to after input file"
    )

    arg_parser.add_argument(
        "-opt",
        help="Optimize the SMTLib program by lowering "
        "pairs and applying constant folding.",
        action="store_true",
    )


def integer_value_refinement(
    value: SSAValue, value_after: SSAValue, insert_point: InsertPoint
) -> SSAValue:
    with ImplicitBuilder(Builder(insert_point)):
        not_after_poison = NotOp.get(SecondOp(value_after).res).res
        value_eq = EqOp.get(FirstOp(value).res, FirstOp(value_after).res).res
        value_refinement = AndOp.get(not_after_poison, value_eq).res
        refinement = OrOp.get(value_refinement, SecondOp(value).res).res
    return refinement


def get_block_ids_from_value(
    val: SSAValue, insert_point: InsertPoint
) -> list[SSAValue]:
    """Get all block ids that are used in the value."""
    if isinstance(val.type, BlockIDType):
        return [val]
    if isattr(val.type, AnyPairType):
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
        block_refinement = ImpliesOp(in_bounds, value_refinement).res
        YieldOp(block_refinement)

    with ImplicitBuilder(Builder(insert_point)):
        size_refinement = EqOp(size, size_after).res
        live_refinement = EqOp(live, live_after).res
        block_properties_refinement = AndOp(size_refinement, live_refinement).res
        block_refinement = AndOp(block_properties_refinement, forall.res).res

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
        refinement = ConstantBoolOp(True).res

    for block_id in accessible_block_ids:
        block_id_after = get_mapped_block_id(
            ret_block_ids_before, ret_block_ids_after, block_id, insert_point
        )
        with ImplicitBuilder(Builder(insert_point)):
            block = mem.GetBlockOp(memory, block_id).res
            block_after = mem.GetBlockOp(memory_after, block_id_after).res

        block_refinement = memory_block_refinement(block, block_after, insert_point)
        with ImplicitBuilder(Builder(insert_point)):
            refinement = AndOp(refinement, block_refinement).res

    return refinement


def add_function_refinement(
    func: DefineFunOp,
    func_after: DefineFunOp,
    function_type: FunctionType,
    insert_point: InsertPoint,
):
    """
    Create operations to check that one function refines another.
    An assert check is added to the end of the list of operations.
    """
    args: list[SSAValue] = []
    builder = Builder(insert_point)
    with ImplicitBuilder(builder):
        # Quantify over all arguments
        for arg in func.body.blocks[0].args:
            const_op = DeclareConstOp(arg.type)
            args.append(const_op.res)

        # Call both operations
        func_call = CallOp(func.ret, args)
        func_call_after = CallOp(func_after.ret, args)

        # Refinement of non-state return values
        return_values_refinement = ConstantBoolOp(True).res

        # Refines each non-state return value
        for (ret, ret_after), original_type in zip(
            zip(func_call.res[:-1], func_call_after.res[:-1], strict=True),
            function_type.outputs.data,
            strict=True,
        ):
            if not isinstance(original_type, IntegerType):
                raise Exception("Cannot handle non-integer return types")
            not_after_poison = NotOp.get(SecondOp(ret_after).res)
            value_eq = EqOp.get(FirstOp(ret).res, FirstOp(ret_after).res)
            value_refinement = AndOp.get(not_after_poison.res, value_eq.res)
            refinement = OrOp.get(value_refinement.res, SecondOp(ret).res)
            return_values_refinement = AndOp.get(
                return_values_refinement, refinement.res
            ).res
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
                NotOp(res_ub_after).res,
                AndOp(return_values_refinement, mem_refinement).res,
            ).res,
            res_ub,
        ).res
        refinement.name_hint = "function_refinement"

        not_refinement = NotOp(refinement).res
        AssertOp(not_refinement)


def remove_effect_states(func: DefineFunOp) -> None:
    effect_state = func.body.blocks[0].args[-1]
    assert len(effect_state.uses) == 1, (
        "xdsl-synth does not handle operations effects yet"
    )
    use = list(effect_state.uses)[0]
    user = use.operation
    assert isinstance(user, ReturnOp), (
        "xdsl-synth does not handle operations with effects yet"
    )
    Rewriter.replace_op(user, ReturnOp(user.ret[:-1]))
    func.body.blocks[0].erase_arg(effect_state)
    assert isinstance(ret := func.ret.type, FunctionType)
    func.ret.type = FunctionType.from_lists(ret.inputs.data[:-1], ret.outputs.data[:-1])


def main() -> None:
    ctx = MLContext()
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(SMTDialect)
    ctx.load_dialect(SMTBitVectorDialect)
    ctx.load_dialect(SMTUtilsDialect)
    ctx.load_dialect(Comb)
    ctx.load_dialect(HW)
    ctx.load_dialect(LLVM)
    ctx.load_dialect(EffectDialect)
    ctx.load_dialect(UBEffectDialect)
    ctx.load_dialect(MemRef)

    # Parse the files
    def parse_file(file: str | None) -> Operation:
        if file is None:
            f = sys.stdin
        else:
            f = open(file)

        parser = Parser(ctx, f.read())
        module = parser.parse_module()
        return module

    module = parse_file(args.before_file)
    module_after = parse_file(args.after_file)

    assert isinstance(module, ModuleOp)
    assert isinstance(module_after, ModuleOp)

    load_vanilla_semantics()

    assert isinstance(module.ops.first, FuncOp)
    func_type = module.ops.first.function_type

    # Convert both module to SMTLib
    LowerToSMTPass().apply(ctx, module)
    LowerToSMTPass().apply(ctx, module_after)

    # Collect the function from both modules
    if (
        len(module.ops) != len(module_after.ops)
        or not isinstance(module.ops.first, DefineFunOp)
        or not isinstance(module_after.ops.first, DefineFunOp)
    ):
        print("Input is expected to have a single `func.func` operation.")
        exit(1)

    func = module.ops.first
    func_after = module_after.ops.first

    # Combine both modules into a new one
    new_module = ModuleOp([])
    block = new_module.body.blocks[0]
    func.detach()
    block.add_op(func)
    func_after.detach()
    block.add_op(func_after)

    # Optionally simplify the module
    if args.opt:
        CanonicalizePass().apply(ctx, new_module)
        CommonSubexpressionElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        # Remove this once we update to latest xdsl
        DeadCodeElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)

    LowerMemoryEffectsPass().apply(ctx, new_module)

    # Optionally simplify the module
    if args.opt:
        CanonicalizePass().apply(ctx, new_module)
        CommonSubexpressionElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        # Remove this once we update to latest xdsl
        DeadCodeElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)

    LowerEffectsWithMemoryPass().apply(ctx, new_module)

    # Optionally simplify the module
    if args.opt:
        CanonicalizePass().apply(ctx, new_module)
        CommonSubexpressionElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        # Remove this once we update to latest xdsl
        DeadCodeElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)

    # Add refinement operations
    add_function_refinement(func, func_after, func_type, InsertPoint.at_end(block))
    block.add_op(CheckSatOp())

    # Inline and delete functions
    FunctionCallInline(True, {}).apply(ctx, new_module)
    for op in new_module.body.ops:
        if isinstance(op, DefineFunOp):
            new_module.body.block.erase_op(op)

    # Optionally simplify the module
    if args.opt:
        LowerPairs().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        CommonSubexpressionElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        # Remove this once we update to latest xdsl
        DeadCodeElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)

    # Lower memory to arrays
    LowerMemoryToArrayPass().apply(ctx, new_module)

    if args.opt:
        CanonicalizePass().apply(ctx, new_module)
        LowerPairs().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        CommonSubexpressionElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)

    print_to_smtlib(new_module, sys.stdout)


if __name__ == "__main__":
    main()
