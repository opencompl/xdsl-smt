#!/usr/bin/env python3

import argparse

from xdsl.ir import MLContext, Operation
from xdsl.parser import Parser

from ..dialects.smt_dialect import SMTDialect
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.arith_dialect import Arith
from ..dialects.index_dialect import Index
from ..dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp
from ..dialects.transfer import Transfer, AbstractValueType

from ..passes import calculate_smt as cs
from ..utils.trans_interpreter_smt import *
from ..passes.rename_values import RenameValuesPass
from ..passes.lower_to_smt.lower_to_smt import LowerToSMT, integer_poison_type_lowerer
from ..passes.pdl_to_smt import PDLToSMT
from ..passes.lower_to_smt import (
    arith_to_smt_patterns,
    func_to_smt_patterns,
    transfer_to_smt_patterns,
)

from z3 import BitVec, Solver, And, Not, simplify, ForAll, Implies
import sys as sys


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

KEY_NEED_VERIFY = "builtin.NEED_VERIFY"
MAXIMAL_VERIFIED_BITS = 8


def main() -> None:
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
    module = parse_file(ctx, args.transfer_functions)
    assert isinstance(module, ModuleOp)

    get_constraint = None
    get_instance_constraint = None

    func_name_to_func = {}

    RenameValuesPass().apply(ctx, module)
    for func in module.ops:
        if isinstance(func, FuncOp):
            func_name_to_func[func.sym_name.data] = func
            if func.sym_name.data == "getConstraint":
                get_constraint = func
            elif func.sym_name.data == "getInstanceConstraint":
                get_instance_constraint = func
    LowerToSMT.rewrite_patterns = [
        *arith_to_smt_patterns,
        *transfer_to_smt_patterns,
        *func_to_smt_patterns,
    ]
    LowerToSMT.type_lowerers = [integer_poison_type_lowerer]
    cloned_op = module.clone()
    PDLToSMT().apply(ctx, cloned_op)

if __name__ == "__main__":
    main()
