#!/usr/bin/env python3

import argparse
import sys

from xdsl.ir import MLContext, Operation
from xdsl.parser import MLIRParser

from dialects.smt_bitvector_dialect import SMTBitVectorDialect
from dialects.smt_dialect import CallOp, DefineFunOp, EqOp, AssertOp, SMTDialect
from dialects.smt_bitvector_dialect import SMTBitVectorDialect
from dialects.arith_dialect import Arith
from dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func
from xdsl.dialects.transfer import Transfer

from passes.lower_pairs import LowerPairs
from passes.arith_to_smt import ArithToSMT
from passes.calculate_smt import CalculateSMT
from passes.canonicalize_smt import CanonicalizeSMT

from traits.smt_printer import print_to_smtlib


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "transfer_functions", type=str, nargs="?", help="path to the transfer functions"
    )

    arg_parser.add_argument(
        "-opt",
        help="Optimize the SMTLib program by lowering "
        "pairs and applying constant folding.",
        action="store_true",
    )


def parse_file(file: str | None) -> Operation:
    if file is None:
        f = sys.stdin
    else:
        f = open(file)

    parser = MLIRParser(ctx, f.read())
    module = parser.parse_op()
    return module


def function_refinement(func: DefineFunOp, func_after: DefineFunOp) -> list[Operation]:
    """
    Create operations to check that one function refines another.
    An assert check is added to the end of the list of operations.
    """
    if len(func.body.blocks[0].args) != 0 or len(func_after.body.blocks[0].args) != 0:
        print("Function with arguments are not yet supported")
        exit(1)

    ops = list[Operation]()

    # Call both operations
    func_call = CallOp.get(func.results[0], [])
    func_call_after = CallOp.get(func_after.results[0], [])
    ops += [func_call, func_call_after]

    # Get the function return value
    ret_val = func_call.res
    ret_val_after = func_call_after.res

    refinement_op = EqOp(ret_val, ret_val_after)
    ops.append(refinement_op)

    ops.append(AssertOp(refinement_op.res))

    return ops


if __name__ == "__main__":
    ctx = MLContext()
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(SMTDialect)
    ctx.register_dialect(SMTBitVectorDialect)
    ctx.register_dialect(SMTUtilsDialect)
    ctx.register_dialect(Transfer)


    # Parse the files
    module = parse_file(args.transfer_functions)
    func0 = module.ops[0]
    func1 = module.ops[1]

    assert isinstance(module, ModuleOp)

    # Convert both module to SMTLib
    CalculateSMT().apply(ctx, func0)
    CalculateSMT().apply(ctx, func1)
    print(module)
    block=func1.body.blocks[0]
    print("smtBegin:")
    for op in block.ops:
        if hasattr(op,"smtVal"):
            print(op)
            print(getattr(op,"smtVal"))
    block = func0.body.blocks[0]
    print("smtBegin:")
    for op in block.ops:
        if hasattr(op, "smtVal"):
            print(op)
            print(getattr(op, "smtVal"))
    exit()

