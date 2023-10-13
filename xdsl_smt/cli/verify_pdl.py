#!/usr/bin/env python3
"""
Verify a PDL rewrite for different integer types.
Check for both correctness and precision.
"""

import subprocess

from io import StringIO
from typing import Iterable
from xdsl.ir import MLContext

from xdsl.dialects.builtin import Builtin, ModuleOp, IntegerType
from xdsl.dialects.func import Func
from xdsl.dialects.pdl import PDL, TypeOp
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.xdsl_opt_main import xDSLOptMain


from ..dialects.hoare_dialect import Hoare
from ..dialects.pdl_dataflow import PDLDataflowDialect
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.smt_dialect import SMTDialect
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.smt_utils_dialect import SMTUtilsDialect
from ..dialects.index_dialect import Index
from ..dialects.transfer import TransIntegerType, Transfer
from ..dialects.hw_dialect import HW
from ..passes.lower_to_smt.lower_to_smt import LowerToSMT, integer_poison_type_lowerer
from ..passes.pdl_to_smt import PDLToSMT
from ..passes.lower_to_smt import (
    arith_to_smt_patterns,
    func_to_smt_patterns,
    transfer_to_smt_patterns,
)
from ..traits.smt_printer import print_to_smtlib

MAX_INT = 32


def verify_pattern(ctx: MLContext, op: ModuleOp) -> bool:
    cloned_op = op.clone()
    PDLToSMT().apply(ctx, cloned_op)
    stream = StringIO()
    print_to_smtlib(cloned_op, stream)

    res = subprocess.run(
        ["z3", "-in"],
        capture_output=True,
        input=stream.getvalue(),
        text=True,
    )
    if res.returncode != 0:
        raise Exception(res.stderr)

    return "unsat" in res.stdout


def iterate_on_all_integers(
    op: ModuleOp, types: tuple[int, ...] = ()
) -> Iterable[tuple[ModuleOp, tuple[int, ...]]]:
    for sub_op in op.walk():
        if isinstance(sub_op, TypeOp) and isinstance(
            sub_op.constantType, TransIntegerType
        ):
            for i in range(1, MAX_INT + 1):
                sub_op.constantType = IntegerType(i)
                sub_types = (*types, i)
                yield from iterate_on_all_integers(op, sub_types)
            return
    yield op, types


class OptMain(xDSLOptMain):
    def register_all_dialects(self):
        self.ctx.register_dialect(Arith)
        self.ctx.register_dialect(Builtin)
        self.ctx.register_dialect(Func)
        self.ctx.register_dialect(Index)
        self.ctx.register_dialect(SMTDialect)
        self.ctx.register_dialect(SMTBitVectorDialect)
        self.ctx.register_dialect(SMTUtilsDialect)
        self.ctx.register_dialect(Transfer)
        self.ctx.register_dialect(Hoare)
        self.ctx.register_dialect(PDL)
        self.ctx.register_dialect(PDLDataflowDialect)
        self.ctx.register_dialect(Comb)
        self.ctx.register_dialect(HW)

    def run(self):
        """Executes the different steps."""
        chunks, file_extension = self.prepare_input()
        assert len(chunks) == 1
        chunk = chunks[0]

        try:
            module = self.parse_chunk(chunk, file_extension)
            assert module is not None
        finally:
            chunk.close()

        for specialized_pattern, types in iterate_on_all_integers(module):
            if verify_pattern(self.ctx, specialized_pattern):
                print(f"with types {types}: SOUND")
            else:
                print(f"with types {types}: UNSOUND")


def main() -> None:
    LowerToSMT.rewrite_patterns = [
        *arith_to_smt_patterns,
        *transfer_to_smt_patterns,
        *func_to_smt_patterns,
    ]
    LowerToSMT.type_lowerers = [integer_poison_type_lowerer]

    OptMain().run()


if __name__ == "__main__":
    main()
