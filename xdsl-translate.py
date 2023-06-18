#!/usr/bin/env python3

import argparse

from xdsl.ir import MLContext, Operation
from xdsl.parser import Parser

from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp, Call
from xdsl_smt.dialects.transfer import Transfer
from xdsl.printer import Printer
from xdsl_smt.passes.transfer_lower import LowerToCpp, addDispatcher

from z3 import *


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "transfer_functions", type=str, nargs="?", help="path to the transfer functions"
    )


def parse_file(file: str | None) -> Operation:
    if file is None:
        f = sys.stdin
    else:
        f = open(file)

    parser = Parser(ctx, f.read(), file)
    module = parser.parse_op()
    return module


if __name__ == "__main__":
    ctx = MLContext()
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(Transfer)

    # Parse the files
    module = parse_file(args.transfer_functions)
    assert isinstance(module, ModuleOp)

    allFuncMapping = {}
    with open("tmp.cpp", "w") as fout:
        for func in module.ops:
            if isinstance(func, FuncOp):
                for op in func.body.blocks[0].ops:
                    pass
                    # print(isinstance(op,Call))
                allFuncMapping[func.sym_name.data] = func
                LowerToCpp(fout).apply(ctx, func)
        addDispatcher(fout)

    # printer = Printer(target=Printer.Target.MLIR)
    # printer.print_op(module)
