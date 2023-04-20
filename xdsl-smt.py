#!/usr/bin/env python3

import argparse
from xdsl.xdsl_opt_main import xDSLOptMain

from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.dialects.pdl import PDL

from dialects.smt_bitvector_dialect import SMTBitVectorDialect
from dialects.smt_dialect import SMTDialect
from dialects.smt_bitvector_dialect import SMTBitVectorDialect
from dialects.arith_dialect import Arith
from dialects.smt_utils_dialect import SMTUtilsDialect

from passes.canonicalize_smt import CanonicalizeSMT
from passes.dead_code_elimination import DeadCodeElimination
from passes.lower_pairs import LowerPairs
from passes.arith_to_smt import ArithToSMT
from passes.pdl_to_smt import PDLToSMT

from traits.smt_printer import print_to_smtlib


class OptMain(xDSLOptMain):

    def register_all_dialects(self):
        self.ctx.register_dialect(Arith)
        self.ctx.register_dialect(Builtin)
        self.ctx.register_dialect(Func)
        self.ctx.register_dialect(SMTDialect)
        self.ctx.register_dialect(SMTBitVectorDialect)
        self.ctx.register_dialect(SMTUtilsDialect)
        self.ctx.register_dialect(PDL)

    def register_all_passes(self):
        super().register_all_passes()
        self.register_pass(ArithToSMT)
        self.register_pass(DeadCodeElimination)
        self.register_pass(CanonicalizeSMT)
        self.register_pass(LowerPairs)
        self.register_pass(PDLToSMT)

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)

    def register_all_targets(self):
        super().register_all_targets()
        self.available_targets['smt'] = print_to_smtlib


def __main__():
    xdsl_main = OptMain()
    xdsl_main.run()


if __name__ == "__main__":
    __main__()