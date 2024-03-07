#!/usr/bin/env python3
"""
Verify a PDL rewrite for different integer types.
Check for both correctness and precision.
"""

import subprocess
import argparse

from io import StringIO
from typing import Iterable
from xdsl.ir import MLContext

from xdsl.dialects.builtin import Builtin, IntegerAttr, ModuleOp, IntegerType
from xdsl.dialects.func import Func
from xdsl.dialects.pdl import PDL, PatternOp, TypeOp
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.xdsl_opt_main import xDSLOptMain

from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.builtin_semantics import IntegerAttrSemantics
from xdsl_smt.semantics.comb_semantics import comb_semantics


from ..dialects.hoare_dialect import Hoare
from ..dialects.pdl_dataflow import PDLDataflowDialect
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.smt_dialect import SMTDialect
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.smt_utils_dialect import SMTUtilsDialect
from ..dialects.index_dialect import Index
from ..dialects.transfer import TransIntegerType, Transfer
from ..dialects.hw_dialect import HW
from ..dialects.llvm_dialect import LLVM

from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl_smt.passes.canonicalize_smt import CanonicalizeSMT
from xdsl_smt.passes.lower_to_smt.lower_to_smt import (
    LowerToSMT,
    integer_poison_type_lowerer,
)
from xdsl_smt.passes.pdl_to_smt import PDLToSMT
from xdsl_smt.passes.lower_to_smt import (
    func_to_smt_patterns,
    transfer_to_smt_patterns,
    llvm_to_smt_patterns,
)
from ..traits.smt_printer import print_to_smtlib
from xdsl_smt.pdl_constraints.integer_arith_constraints import (
    integer_arith_native_rewrites,
    integer_arith_native_constraints,
    integer_arith_native_static_constraints,
)

max_bitwidth = 32


def verify_pattern(ctx: MLContext, op: ModuleOp, opt: bool) -> bool:
    cloned_op = op.clone()
    PDLToSMT().apply(ctx, cloned_op)
    if opt:
        LowerPairs().apply(ctx, cloned_op)
        CanonicalizeSMT().apply(ctx, cloned_op)
    cloned_op.verify()
    stream = StringIO()
    print_to_smtlib(cloned_op, stream)

    res = subprocess.run(
        ["z3", "-in"],
        capture_output=True,
        input=stream.getvalue(),
        text=True,
    )
    if res.returncode != 0:
        raise Exception(
            "An exception was raised in the following program: "
            f"{stream.getvalue()} \n\n Error message: {res.stderr}"
        )

    return "unsat" in res.stdout


def iterate_on_all_integers(
    op: PatternOp, types: tuple[int, ...] = ()
) -> Iterable[tuple[PatternOp, tuple[int, ...]]]:
    op = op.clone()
    for sub_op in op.walk():
        if isinstance(sub_op, TypeOp) and isinstance(
            sub_op.constantType, TransIntegerType
        ):
            for i in range(1, max_bitwidth + 1):
                sub_op.constantType = IntegerType(i)
                sub_types = (*types, i)
                yield from iterate_on_all_integers(op, sub_types)
            return
    yield op, types


class OptMain(xDSLOptMain):
    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        arg_parser.add_argument(
            "-max-bitwidth",
            type=int,
            default=32,
            help="maximum bitwidth of integer types",
        )
        arg_parser.add_argument(
            "-opt",
            type=bool,
            default=False,
            action="store_true",
            help="Optimize the SMT query before sending it to Z3",
        )

        super().register_all_arguments(arg_parser)

    def register_all_dialects(self):
        self.ctx.load_dialect(Arith)
        self.ctx.load_dialect(Builtin)
        self.ctx.load_dialect(Func)
        self.ctx.load_dialect(Index)
        self.ctx.load_dialect(SMTDialect)
        self.ctx.load_dialect(SMTBitVectorDialect)
        self.ctx.load_dialect(SMTUtilsDialect)
        self.ctx.load_dialect(Transfer)
        self.ctx.load_dialect(Hoare)
        self.ctx.load_dialect(PDL)
        self.ctx.load_dialect(PDLDataflowDialect)
        self.ctx.load_dialect(Comb)
        self.ctx.load_dialect(HW)
        self.ctx.load_dialect(LLVM)

    def run(self):
        """Executes the different steps."""

        global max_bitwidth
        max_bitwidth = self.args.max_bitwidth

        chunks, file_extension = self.prepare_input()
        assert len(chunks) == 1
        chunk = chunks[0]

        try:
            module = self.parse_chunk(chunk, file_extension)
            assert module is not None
        finally:
            chunk.close()

        is_one_unsound = False

        for pattern in module.walk():
            if isinstance(pattern, PatternOp):
                if pattern.sym_name:
                    print(f"Verifying pattern {pattern.sym_name.data}:")
                else:
                    print(f"Verifying pattern:")
                for specialized_pattern, types in iterate_on_all_integers(pattern):
                    if verify_pattern(
                        self.ctx, ModuleOp([specialized_pattern]), self.args.opt
                    ):
                        print(f"with types {types}: SOUND")
                    else:
                        print(f"with types {types}: UNSOUND")
                        is_one_unsound = True

        if is_one_unsound:
            print("At least one pattern is unsound")
        else:
            print("All patterns are sound")


def main() -> None:
    LowerToSMT.rewrite_patterns = [
        *transfer_to_smt_patterns,
        *func_to_smt_patterns,
        *llvm_to_smt_patterns,
    ]
    LowerToSMT.type_lowerers = [integer_poison_type_lowerer]
    LowerToSMT.attribute_semantics = {IntegerAttr: IntegerAttrSemantics()}
    LowerToSMT.operation_semantics = {**arith_semantics, **comb_semantics}

    PDLToSMT.native_rewrites = integer_arith_native_rewrites
    PDLToSMT.native_constraints = integer_arith_native_constraints
    PDLToSMT.native_static_constraints = integer_arith_native_static_constraints

    OptMain().run()


if __name__ == "__main__":
    main()
