#!/usr/bin/env python3

import argparse

from xdsl.ir import MLContext, Operation
from xdsl.parser import MLIRParser

from dialects.smt_dialect import SMTDialect
from dialects.smt_bitvector_dialect import SMTBitVectorDialect
from dialects.arith_dialect import Arith
from dialects.index_dialect import Index
from dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.transfer import Transfer
from xdsl.printer import Printer
from passes.knownBitsAnalysis import KnownBitsAnalysisPass

from z3 import *


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "transfer_functions", type=str, nargs="?", help="path to the transfer functions"
    )

    arg_parser.add_argument(
        "analysis_file", type=str, nargs="?", help="path to the transfer functions"
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
    ctx.register_dialect(Index)

    # Parse the files
    module = parse_file(args.transfer_functions)
    analysis_file=parse_file(args.analysis_file)
    assert isinstance(module, ModuleOp)
    assert isinstance(analysis_file, ModuleOp)

    transferFuncMapping={}
    for func in module.ops:
        if isinstance(func, FuncOp):
            if "applied_to" in func.attributes:
                for funcName in func.attributes["applied_to"]:
                    transferFuncMapping[funcName.data]=func



    for func in analysis_file.ops:
        if isinstance(func,FuncOp):
            KnownBitsAnalysisPass(transferFuncMapping).apply(ctx,func)
    printer = Printer(target=Printer.Target.MLIR)
    printer.print_op(analysis_file)