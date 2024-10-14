#!/usr/bin/env python3

import argparse
import sys

from xdsl.context import MLContext
from xdsl.ir import Operation, SSAValue
from xdsl.parser import Parser
from xdsl.rewriter import Rewriter
from xdsl.utils.hints import isa

from xdsl_smt.passes.lower_to_smt.lower_to_smt import SMTLowerer

from xdsl_smt.dialects import smt_utils_dialect as smt_utils
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import (
    AndOp,
    CallOp,
    CheckSatOp,
    DeclareConstOp,
    DefineFunOp,
    EqOp,
    AssertOp,
    NotOp,
    OrOp,
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
)
from xdsl.dialects.func import Func
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb

from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl_smt.passes.canonicalize_smt import CanonicalizeSMT
from xdsl_smt.passes.lower_to_smt import (
    LowerToSMTPass,
    func_to_smt_patterns,
)
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.comb_semantics import comb_semantics
from xdsl_smt.semantics.builtin_semantics import (
    IndexTypeSemantics,
    IntegerTypeSemantics,
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


def function_refinement(func: DefineFunOp, func_after: DefineFunOp) -> list[Operation]:
    """
    Create operations to check that one function refines another.
    An assert check is added to the end of the list of operations.
    """
    args: list[SSAValue] = []
    ops = list[Operation]()

    for arg in func.body.blocks[0].args:
        const_op = DeclareConstOp(arg.type)
        ops.append(const_op)
        args.append(const_op.res)

    # Call both operations
    func_call = CallOp.get(func.results[0], args)
    func_call_after = CallOp.get(func_after.results[0], args)
    ops += [func_call, func_call_after]

    # Get the function return values and poison
    ret_value = FirstOp(func_call.res)
    ret_poison = SecondOp(func_call.res)
    ops.extend((ret_value, ret_poison))

    ret_value_after = FirstOp(func_call_after.res)
    ret_poison_after = SecondOp(func_call_after.res)
    ops.extend((ret_value_after, ret_poison_after))

    not_after_poison = NotOp.get(ret_poison_after.res)
    value_eq = EqOp.get(ret_value.res, ret_value_after.res)
    value_refinement = AndOp.get(not_after_poison.res, value_eq.res)
    refinement = OrOp.get(value_refinement.res, ret_poison.res)
    ops.extend([not_after_poison, value_eq, value_refinement, refinement])

    not_refinement = NotOp.get(refinement.res)
    ops.append(not_refinement)

    assert_op = AssertOp(not_refinement.res)
    ops.append(assert_op)

    return ops


def remove_effect_states(func: DefineFunOp) -> None:
    effect_state = func.body.blocks[0].args[-1]
    assert len(effect_state.uses) == 1, "xdsl-synth does not handle effects yet"
    user = list(effect_state.uses)[0].operation
    assert isinstance(user, smt_utils.PairOp)
    Rewriter.replace_op(user, [], [user.first])
    func.body.blocks[0].erase_arg(effect_state)
    assert isinstance(ret := func.ret.type, FunctionType)
    assert isa(pair := ret.outputs.data[0], smt_utils.AnyPairType)
    func.ret.type = FunctionType.from_lists(ret.inputs.data[:-1], [pair.first])


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
    }
    SMTLowerer.op_semantics = {**arith_semantics, **comb_semantics}

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

    # HACK: As the effect system is still wip, we do not handle effectse here yet
    remove_effect_states(func)
    remove_effect_states(func_after)

    # Combine both modules into a new one
    new_module = ModuleOp([])
    block = new_module.body.blocks[0]
    func.detach()
    block.add_op(func)
    func_after.detach()
    block.add_op(func_after)

    # Add refinement operations
    block.add_ops(function_refinement(func, func_after))
    block.add_op(CheckSatOp())

    if args.opt:
        LowerPairs().apply(ctx, new_module)
        CanonicalizeSMT().apply(ctx, new_module)
    print_to_smtlib(new_module, sys.stdout)


if __name__ == "__main__":
    main()
