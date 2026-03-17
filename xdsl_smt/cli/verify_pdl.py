#!/usr/bin/env python3
"""
Verify a PDL rewrite for different integer types.
Check for both correctness and precision.
"""

import subprocess
import argparse

from io import StringIO
from typing import Iterable
from xdsl.ir import Dialect
from xdsl.context import Context

from xdsl.dialects.builtin import Builtin, ModuleOp, IntegerType, SymbolRefAttr
from xdsl.dialects.func import Func
from xdsl.dialects.pdl import PDL, PatternOp, TypeOp
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)
from xdsl.xdsl_opt_main import xDSLOptMain

from xdsl_smt.passes.lower_effects import LowerEffectPass

from ..dialects.hoare_dialect import Hoare
from ..dialects.pdl_dataflow import PDLDataflowDialect
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.smt_dialect import SMTDialect
from ..dialects.smt_utils_dialect import SMTUtilsDialect
from ..dialects.index_dialect import Index
from ..dialects.transfer import TransIntegerType, Transfer
from ..dialects.hw_dialect import HW
from ..dialects.llvm_dialect import LLVM

from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl_smt.passes.pdl_to_smt import PDLToSMT
from ..traits.smt_printer import print_to_smtlib
from xdsl_smt.pdl_constraints.integer_arith_constraints import (
    integer_arith_native_rewrites,
    integer_arith_native_constraints,
    integer_arith_native_static_constraints,
)
from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import load_vanilla_semantics
from xdsl_smt.passes.smt_expand import SMTExpand


def verify_pattern(ctx: Context, op: ModuleOp, opt: bool) -> bool:
    cloned_op = op.clone()
    PDLToSMT().apply(ctx, cloned_op)
    LowerEffectPass().apply(ctx, cloned_op)
    SMTExpand().apply(ctx, cloned_op)
    if opt:
        LowerPairs().apply(ctx, cloned_op)
        CanonicalizePass().apply(ctx, cloned_op)
        CommonSubexpressionElimination().apply(ctx, cloned_op)
        CanonicalizePass().apply(ctx, cloned_op)
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
    op: PatternOp,
    max_bitwidth: int,
) -> Iterable[tuple[PatternOp, tuple[int, ...]]]:
    # Find all the TypeOp in the pattern
    type_ops: list[TypeOp] = []
    for sub_op in op.walk():
        if isinstance(sub_op, TypeOp) and isinstance(
            sub_op.constantType, TransIntegerType
        ):
            type_ops.append(sub_op)

    # No type to specialize case
    if not type_ops:
        yield op.clone(), ()
        return

    # The initial types are all 1
    bitwidths = [1 for _ in type_ops]
    while True:
        # Assign the types in the pattern:
        for type_op, bitwidth in zip(type_ops, bitwidths):
            type_op.constantType = IntegerType(bitwidth)

        yield op.clone(), tuple(bitwidths)

        # Get the next bitwidths to try
        index = 0
        while index < len(bitwidths):
            bitwidth = bitwidths[index]
            if bitwidth == max_bitwidth:
                bitwidths[index] = 1
                index += 1
            else:
                bitwidths[index] += 1
                break
        else:
            break

    for type_op in type_ops:
        type_op.constantType = TransIntegerType(SymbolRefAttr("W"))


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
            default=False,
            action="store_true",
            help="Optimize the SMT query before sending it to Z3",
        )

        super().register_all_arguments(arg_parser)

    def register_all_dialects(self):
        NEW_PDL = Dialect(
            "pdl",
            [*PDL.operations, *PDLDataflowDialect.operations],
            [*PDL.attributes, *PDLDataflowDialect.attributes],
        )
        SMT_COLLECTION = Dialect(
            "smt",
            [
                *SMTDialect.operations,
                *SMTBitVectorDialect.operations,
                *SMTUtilsDialect.operations,
            ],
            [
                *SMTDialect.attributes,
                *SMTBitVectorDialect.attributes,
                *SMTUtilsDialect.attributes,
            ],
        )
        self.ctx.register_dialect(Arith.name, lambda: Arith)
        self.ctx.register_dialect(Builtin.name, lambda: Builtin)
        self.ctx.register_dialect(Func.name, lambda: Func)
        self.ctx.register_dialect(Index.name, lambda: Index)
        self.ctx.register_dialect(SMT_COLLECTION.name, lambda: SMT_COLLECTION)
        self.ctx.register_dialect(Transfer.name, lambda: Transfer)
        self.ctx.register_dialect(Hoare.name, lambda: Hoare)
        self.ctx.register_dialect(PDL.name, lambda: NEW_PDL)
        self.ctx.register_dialect(Comb.name, lambda: Comb)
        self.ctx.register_dialect(HW.name, lambda: HW)
        self.ctx.register_dialect(LLVM.name, lambda: LLVM)

    def run(self):
        """Executes the different steps."""

        chunks, file_extension = self.prepare_input()
        assert len(chunks) == 1
        chunk = chunks[0][0]

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
                for specialized_pattern, types in iterate_on_all_integers(
                    pattern, self.args.max_bitwidth
                ):
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
    load_vanilla_semantics()
    PDLToSMT.pdl_lowerer.native_rewrites = integer_arith_native_rewrites
    PDLToSMT.pdl_lowerer.native_constraints = integer_arith_native_constraints
    PDLToSMT.pdl_lowerer.native_static_constraints = (
        integer_arith_native_static_constraints
    )
    OptMain().run()


if __name__ == "__main__":
    main()
