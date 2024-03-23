#!/usr/bin/env python3

import argparse
import sys

from xdsl.ir import MLContext, Operation, SSAValue
from xdsl.parser import Parser

from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.smt_dialect import (
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
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.smt_utils_dialect import FirstOp, SMTUtilsDialect, SecondOp
from ..dialects.hw_dialect import HW
from ..dialects.llvm_dialect import LLVM
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb

from ..passes.lower_pairs import LowerPairs
from ..passes.canonicalize_smt import CanonicalizeSMT
from ..passes.lower_to_smt import (
    LowerToSMT,
    transfer_to_smt_patterns,
    integer_poison_type_lowerer,
    func_to_smt_patterns,
    llvm_to_smt_patterns,
)
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.comb_semantics import comb_semantics
from ..traits.smt_printer import print_to_smtlib


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

    # Parse the files
    def parse_file(file: str | None) -> Operation:
        if file is None:
            f = sys.stdin
        else:
            f = open(file)

        parser = Parser(ctx, f.read())
        module = parser.parse_op()
        return module

    module = parse_file(args.before_file)
    module_after = parse_file(args.after_file)

    assert isinstance(module, ModuleOp)
    assert isinstance(module_after, ModuleOp)

    LowerToSMT.rewrite_patterns = [
        *transfer_to_smt_patterns,
        *func_to_smt_patterns,
        *llvm_to_smt_patterns,
    ]
    LowerToSMT.type_lowerers = [integer_poison_type_lowerer]
    LowerToSMT.operation_semantics = {**arith_semantics, **comb_semantics}

    # Convert both module to SMTLib
    LowerToSMT().apply(ctx, module)
    LowerToSMT().apply(ctx, module_after)

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

    # Add refinement operations
    block.add_ops(function_refinement(func, func_after))
    block.add_op(CheckSatOp())

    if args.opt:
        LowerPairs().apply(ctx, new_module)
        CanonicalizeSMT().apply(ctx, new_module)
    print_to_smtlib(new_module, sys.stdout)


if __name__ == "__main__":
    main()
