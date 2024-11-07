#!/usr/bin/env python3

import argparse
import sys

from xdsl.context import MLContext
from xdsl.ir import Operation, SSAValue
from xdsl.parser import Parser
from xdsl.rewriter import Rewriter, InsertPoint
from xdsl.builder import ImplicitBuilder, Builder

from xdsl_smt.passes.dead_code_elimination import DeadCodeElimination
from xdsl_smt.passes.lower_effects_with_memory import LowerEffectWithMemoryPass
from xdsl_smt.passes.lower_memory_to_array import LowerMemoryToArrayPass
from xdsl_smt.passes.lower_to_smt.lower_to_smt import SMTLowerer

from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import (
    AndOp,
    CallOp,
    CheckSatOp,
    ConstantBoolOp,
    DeclareConstOp,
    DefineFunOp,
    EqOp,
    AssertOp,
    NotOp,
    OrOp,
    ReturnOp,
    SMTDialect,
)
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.effects.ub_effect import UBEffectDialect
from xdsl_smt.dialects.effects.effect import EffectDialect
from xdsl_smt.dialects.smt_utils_dialect import FirstOp, SMTUtilsDialect, SecondOp
from xdsl_smt.dialects.hw_dialect import HW
from xdsl_smt.dialects.llvm_dialect import LLVM
from xdsl.dialects.builtin import (
    Builtin,
    ModuleOp,
    IntegerType,
    IndexType,
    FunctionType,
    MemRefType,
)
from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.dialects.memref import MemRef

from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl_smt.passes.lower_to_smt import (
    LowerToSMTPass,
    func_to_smt_patterns,
)
from xdsl_smt.passes.transfer_inline import FunctionCallInline
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.comb_semantics import comb_semantics
from xdsl_smt.semantics.builtin_semantics import (
    IndexTypeSemantics,
    IntegerTypeSemantics,
)
from xdsl_smt.semantics.memref_semantics import memref_semantics, MemrefSemantics
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

        # Get ub results
        res_ub = SecondOp(func_call.res[-1]).res
        res_ub_after = SecondOp(func_call_after.res[-1]).res

        # Compute refinement with UB
        refinement = OrOp(
            AndOp(NotOp(res_ub_after).res, return_values_refinement).res, res_ub
        ).res

        not_refinement = NotOp(refinement).res
        AssertOp(not_refinement)


def remove_effect_states(func: DefineFunOp) -> None:
    effect_state = func.body.blocks[0].args[-1]
    assert (
        len(effect_state.uses) == 1
    ), "xdsl-synth does not handle operations effects yet"
    use = list(effect_state.uses)[0]
    user = use.operation
    assert isinstance(
        user, ReturnOp
    ), "xdsl-synth does not handle operations with effects yet"
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

    SMTLowerer.rewrite_patterns = {
        # *transfer_to_smt_patterns,
        **func_to_smt_patterns,
        # *llvm_to_smt_patterns,
    }
    SMTLowerer.type_lowerers = {
        IntegerType: IntegerTypeSemantics(),
        IndexType: IndexTypeSemantics(),
        MemRefType: MemrefSemantics(),
    }
    SMTLowerer.op_semantics = {**arith_semantics, **comb_semantics, **memref_semantics}

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

    LowerEffectWithMemoryPass().apply(ctx, new_module)

    # Optionally simplify the module
    if args.opt:
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

    # Lower memory to arrays
    LowerMemoryToArrayPass().apply(ctx, new_module)

    if args.opt:
        LowerPairs().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
    print_to_smtlib(new_module, sys.stdout)


if __name__ == "__main__":
    main()
