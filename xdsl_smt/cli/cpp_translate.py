#!/usr/bin/env python3

import argparse

from xdsl.ir import MLContext, Operation
from xdsl.parser import Parser

from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp, IntegerAttr
from xdsl.dialects.func import Func, FuncOp, Call
from ..dialects.transfer import Transfer
from ..dialects.llvm_dialect import LLVM
from xdsl.printer import Printer
from ..passes.transfer_lower import LowerToCpp, addDispatcher, addInductionOps

from z3 import *


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "transfer_functions", type=str, nargs="?", help="path to the transfer functions"
    )


def parse_file(ctx, file: str | None) -> Operation:
    if file is None:
        f = sys.stdin
    else:
        f = open(file)

    parser = Parser(ctx, f.read(), file)
    module = parser.parse_op()
    return module

def is_transfer_function(func:FuncOp) -> bool:
    return "applied_to" in func.attributes

def is_forward(func:FuncOp) -> bool:
    if "is_forward" in func.attributes:
        forward = func.attributes['is_forward']
        assert isinstance(forward, IntegerAttr)
        return forward.value.data == 1
    return False

def main() -> None:
    ctx = MLContext()
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(Transfer)
    ctx.load_dialect(LLVM)

    # Parse the files
    module = parse_file(ctx, args.transfer_functions)
    assert isinstance(module, ModuleOp)

    allFuncMapping = {}
    forward=False
    with open("tmp.cpp", "w") as fout:
        LowerToCpp.fout = fout
        for func in module.ops:
            if isinstance(func, FuncOp):
                for op in func.body.blocks[0].ops:
                    pass
                    # print(isinstance(op,Call))
                allFuncMapping[func.sym_name.data] = func
                LowerToCpp().apply(ctx, func)
                forward |= (is_transfer_function(func) and is_forward(func))

        addInductionOps(fout)
        addDispatcher(fout, forward)

    # printer = Printer(target=Printer.Target.MLIR)
    # printer.print_op(module)


if __name__ == "__main__":
    main()
