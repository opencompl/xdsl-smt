#!/usr/bin/env python3

import argparse
import sys

from xdsl.context import Context
from xdsl.ir import Operation
from xdsl.parser import Parser
from xdsl.rewriter import InsertPoint
from xdsl_smt.passes.dead_code_elimination import DeadCodeElimination
from xdsl_smt.passes.lower_effects_with_memory import (
    LowerEffectsWithMemoryPass,
)
from xdsl_smt.passes.lower_memory_effects import LowerMemoryEffectsPass
from xdsl_smt.passes.lower_memory_to_array import LowerMemoryToArrayPass
from xdsl_smt.passes.smt_expand import SMTExpand

from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import (
    AssertOp,
    CheckSatOp,
    DefineFunOp,
    NotOp,
    SMTDialect,
)
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.effects.ub_effect import UBEffectDialect
from xdsl_smt.dialects.effects.effect import EffectDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl_smt.dialects.hw_dialect import HW
from xdsl_smt.dialects.llvm_dialect import LLVM
from xdsl.dialects.builtin import (
    Builtin,
    ModuleOp,
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
from xdsl_smt.semantics.refinements import (
    insert_function_refinement_with_declare_const,
)
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


def main() -> None:
    ctx = Context()
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
    func_type_before = module.ops.first.function_type
    assert isinstance(module_after.ops.first, FuncOp)
    func_type_after = module_after.ops.first.function_type

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

    refinement = insert_function_refinement_with_declare_const(
        func,
        func_type_before,
        func_after,
        func_type_after,
        InsertPoint.at_end(block),
    )
    not_op = NotOp(refinement)
    block.add_op(not_op)
    block.add_op(AssertOp(not_op.result))
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

    # Expand ops not supported by all SMT solvers
    SMTExpand().apply(ctx, new_module)
    if args.opt:
        CanonicalizePass().apply(ctx, new_module)
        LowerPairs().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        CommonSubexpressionElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)

    print_to_smtlib(new_module, sys.stdout)


if __name__ == "__main__":
    main()
