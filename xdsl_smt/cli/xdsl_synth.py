#!/usr/bin/env python3

"""
Given an input program and another input program with holes, synthesize
constants that makes the second program a refinement of the first one.
"""

import argparse
import sys
from typing import Sequence

from xdsl.ir import Attribute, BlockArgument, Operation, SSAValue
from xdsl.context import MLContext
from xdsl.parser import Parser
from xdsl.utils.hints import isa
from xdsl.rewriter import Rewriter, InsertPoint

from xdsl_smt.dialects import synth_dialect

from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.smt_dialect import (
    AndOp,
    CallOp,
    CheckSatOp,
    DefineFunOp,
    ReturnOp,
    EqOp,
    AssertOp,
    ForallOp,
    NotOp,
    OrOp,
    SMTDialect,
    YieldOp,
)
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_utils_dialect import (
    FirstOp,
    SMTUtilsDialect,
    SecondOp,
)
from xdsl_smt.dialects.hw_dialect import HW
from xdsl_smt.dialects.llvm_dialect import LLVM
from xdsl_smt.dialects.synth_dialect import SMTSynthDialect
from xdsl.dialects.builtin import Builtin, ModuleOp, FunctionType
from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.builder import Builder, ImplicitBuilder

from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl_smt.passes.lower_to_smt import LowerToSMTPass
from ..traits.smt_printer import print_to_smtlib
from xdsl_smt.dialects import smt_bitvector_dialect

from xdsl_smt.dialects import smt_dialect

from xdsl_smt.dialects import smt_utils_dialect
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


def insert_function_refinement(
    func: DefineFunOp,
    func_after: DefineFunOp,
    builder: Builder,
) -> None:
    """
    Create operations to check that one function refines another.
    An assert check is added to the end of the list of operations.
    """

    synth_types = func_after.func_type.inputs.data[len(func.func_type.inputs.data) :]
    synth_values: list[SSAValue] = []

    with ImplicitBuilder(builder):
        for type in synth_types:
            # Current hack to make this work for poison values
            assert isa(
                type,
                smt_utils_dialect.PairType[
                    smt_bitvector_dialect.BitVectorType, smt_dialect.BoolType
                ],
            )
            const_value = smt_dialect.DeclareConstOp(type.first).res
            no_poison = smt_dialect.ConstantBoolOp(False).res
            synth_values.append(smt_utils_dialect.PairOp(const_value, no_poison).res)

    with ImplicitBuilder(builder):
        forall = ForallOp.from_variables([arg.type for arg in func.body.blocks[0].args])
        AssertOp(forall.res)

    args = forall.regions[0].block.args
    with ImplicitBuilder(forall.regions[0].block):
        # Call both operations
        func_call = CallOp.get(func.results[0], args)
        func_call_after = CallOp.get(func_after.results[0], [*args, *synth_values])

        # Get the function return values and poison
        ret_value = FirstOp(func_call.res[0])
        ret_poison = SecondOp(func_call.res[0])
        ret_value_after = FirstOp(func_call_after.res[0])
        ret_poison_after = SecondOp(func_call_after.res[0])

        # Check for refinement
        not_after_poison = NotOp.get(ret_poison_after.res)
        value_eq = EqOp.get(ret_value.res, ret_value_after.res)
        value_refinement = AndOp.get(not_after_poison.res, value_eq.res)
        refinement = OrOp.get(value_refinement.res, ret_poison.res)

        YieldOp(refinement.res)


def move_synth_constants_to_arguments(func: FuncOp) -> Sequence[Attribute]:
    """Move smt.synth.constant operations to the end of the function arguments."""
    synth_ops: list[synth_dialect.ConstantOp] = []
    arg_types: list[Attribute] = []
    block_arguments: list[BlockArgument] = []

    for op in func.walk():
        if isinstance(op, synth_dialect.ConstantOp):
            synth_ops.append(op)
            arg_types.append(op.res.type)
            block_arguments.append(
                func.body.blocks[0].insert_arg(op.res.type, len(func.args))
            )

    for op, value in zip(synth_ops, block_arguments):
        op.res.replace_by(value)
        op.detach()
        op.erase()

    func.update_function_type()

    return arg_types


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
    ctx.load_dialect(SMTSynthDialect)
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

    # Move smt.synth.constant to function arguments
    func_after = module_after.ops.first
    assert isinstance(func_after, FuncOp)
    move_synth_constants_to_arguments(func_after)

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

    # Remove effect states.
    # This is a temporary hack, and only there because the effect system is still WIP.
    # Once it is ready, we should remove this and add it to the refinement check.
    remove_effect_states(func)
    remove_effect_states(func_after)

    # Add refinement operations
    builder = Builder(InsertPoint.at_end(block))
    insert_function_refinement(func, func_after, builder)
    block.add_op(CheckSatOp())

    if args.opt:
        LowerPairs().apply(ctx, new_module)
        CanonicalizePass().apply(ctx, new_module)
    print_to_smtlib(new_module, sys.stdout)


if __name__ == "__main__":
    main()
