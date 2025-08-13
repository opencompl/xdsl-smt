#!/usr/bin/env python3

import argparse
import sys
from typing import Mapping, Sequence

from xdsl.ir import Attribute, Operation, SSAValue, Region, Block
from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)
from xdsl.rewriter import InsertPoint

from xdsl_smt.dialects import synth_dialect

from xdsl_smt.passes.lower_effects_with_memory import LowerEffectsWithMemoryPass
from xdsl_smt.passes.lower_memory_effects import LowerMemoryEffectsPass
from xdsl_smt.passes.lower_memory_to_array import LowerMemoryToArrayPass
from xdsl_smt.passes.lower_to_smt.smt_lowerer import SMTLowerer
from xdsl_smt.passes.smt_expand import SMTExpand
from xdsl_smt.passes.transfer_inline import FunctionCallInline
from xdsl_smt.semantics.refinements import add_function_refinement
from xdsl_smt.semantics.semantics import OperationSemantics
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.smt_dialect import (
    CheckSatOp,
    DeclareConstOp,
    DefineFunOp,
    AssertOp,
    ForallOp,
    SMTDialect,
    YieldOp,
)
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_utils_dialect import (
    SMTUtilsDialect,
)
from xdsl_smt.dialects.hw_dialect import HW
from xdsl_smt.dialects.llvm_dialect import LLVM
from xdsl_smt.dialects.synth_dialect import SynthDialect
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.builder import Builder

from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl_smt.passes.lower_to_smt import LowerToSMTPass
from ..traits.smt_printer import print_to_smtlib

from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import load_vanilla_semantics


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


def move_synth_constants_outside_of_function(
    func: FuncOp, insert_point: InsertPoint
) -> None:
    """Move synth.constant operations to the beginning of the module."""
    builder = Builder(insert_point)

    for op in func.walk():
        if isinstance(op, synth_dialect.ConstantOp):
            op.detach()
            builder.insert(op)


class SynthSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        result_type = SMTLowerer.lower_type(results[0])
        res = rewriter.insert(DeclareConstOp(result_type)).res
        return ((res,), effect_state)


def main() -> None:
    ctx = Context()
    ctx.allow_unregistered = True
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
    ctx.load_dialect(SynthDialect)
    ctx.load_dialect(Comb)
    ctx.load_dialect(HW)
    ctx.load_dialect(LLVM)

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
    SMTLowerer.op_semantics[synth_dialect.ConstantOp] = SynthSemantics()

    # Move smt.synth.constant to function arguments
    func_after = module_after.ops.first
    assert isinstance(func_after, FuncOp)
    func_type = func_after.function_type
    move_synth_constants_outside_of_function(
        func_after, InsertPoint.at_start(module_after.body.blocks[0])
    )

    # Convert both module to SMTLib
    LowerToSMTPass().apply(ctx, module)
    LowerToSMTPass().apply(ctx, module_after)

    # Collect the function from both modules
    if not isinstance(module.ops.last, DefineFunOp) or not isinstance(
        module_after.ops.last, DefineFunOp
    ):
        print("Input is expected to have a single `func.func` operation.")
        exit(1)

    func = module.ops.last
    func_after = module_after.ops.last

    # Combine both modules into a new one
    new_module = ModuleOp([])
    block = new_module.body.blocks[0]
    for op in module.body.ops:
        op.detach()
        block.add_op(op)
    for op in module_after.body.ops:
        op.detach()
        block.add_op(op)

    if args.opt:
        CanonicalizePass().apply(ctx, new_module)
        LowerPairs().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        CommonSubexpressionElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)

    LowerMemoryEffectsPass().apply(ctx, new_module)

    if args.opt:
        CanonicalizePass().apply(ctx, new_module)
        LowerPairs().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        CommonSubexpressionElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)

    LowerEffectsWithMemoryPass().apply(ctx, new_module)

    if args.opt:
        CanonicalizePass().apply(ctx, new_module)
        LowerPairs().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        CommonSubexpressionElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)

    forall = ForallOp(Region(Block(arg_types=func_after.body.block.arg_types)))
    refinement = add_function_refinement(
        func,
        func_after,
        func_type,
        InsertPoint.at_end(forall.body.block),
        args=forall.body.block.args,
    )
    forall.body.block.add_op(YieldOp(refinement))
    block.add_op(forall)
    block.add_op(AssertOp(forall.result))

    if args.opt:
        CanonicalizePass().apply(ctx, new_module)
        LowerPairs().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        CommonSubexpressionElimination().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)

    FunctionCallInline(True, {}).apply(ctx, new_module)
    for op in new_module.body.ops:
        if isinstance(op, DefineFunOp):
            new_module.body.block.erase_op(op)

    if args.opt:
        CanonicalizePass().apply(ctx, new_module)
        LowerPairs().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
        CommonSubexpressionElimination().apply(ctx, new_module)
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

    block.add_op(CheckSatOp())
    print_to_smtlib(new_module, sys.stdout)


if __name__ == "__main__":
    main()
