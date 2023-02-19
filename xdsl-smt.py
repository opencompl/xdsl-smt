#!/usr/bin/env python3

import argparse
from xdsl.xdsl_opt_main import xDSLOptMain

from xdsl.dialects.builtin import Builtin
from xdsl.dialects.arith import Arith
from xdsl.dialects.func import Func

from dialects.smt_bitvector_dialect import SMTBitVectorDialect
from dialects.smt_dialect import SMTDialect
from dialects.smt_printer_interface import print_to_smtlib
from dialects.smt_bitvector_dialect import SMTBitVectorDialect
from dialects.smt_utils_dialect import SMTUtilsDialect
from passes.arith_to_smt import arith_to_smt


class OptMain(xDSLOptMain):

    def register_all_dialects(self):
        self.ctx.register_dialect(Arith)
        self.ctx.register_dialect(Builtin)
        self.ctx.register_dialect(Func)
        self.ctx.register_dialect(SMTDialect)
        self.ctx.register_dialect(SMTBitVectorDialect)
        self.ctx.register_dialect(SMTUtilsDialect)

    def register_all_passes(self):
        super().register_all_passes()
        self.available_passes['arith_to_smt'] = arith_to_smt

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