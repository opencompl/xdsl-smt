#!/usr/bin/env python3

from xdsl.ir import Dialect
from xdsl.xdsl_opt_main import xDSLOptMain

from xdsl.dialects.builtin import Builtin, IntegerAttr
from xdsl.dialects.func import Func
from xdsl.dialects.pdl import PDL
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl_smt.dialects.smt_ub_dialect import SMTUBDialect
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.builtin_semantics import IntegerAttrSemantics

from xdsl_smt.passes.lower_to_smt import integer_poison_type_lowerer


from ..dialects.hoare_dialect import Hoare
from ..dialects.pdl_dataflow import PDLDataflowDialect
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.smt_dialect import SMTDialect
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.smt_utils_dialect import SMTUtilsDialect
from ..dialects.index_dialect import Index
from ..dialects.transfer import Transfer
from ..dialects.hw_dialect import HW
from ..dialects.llvm_dialect import LLVM

from xdsl_smt.passes.canonicalize_smt import CanonicalizeSMT
from xdsl_smt.passes.dead_code_elimination import DeadCodeElimination
from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl_smt.passes.lower_to_smt import LowerToSMT
from xdsl_smt.semantics.comb_semantics import comb_semantics
from ..passes.pdl_to_smt import PDLToSMT

from ..traits.smt_printer import print_to_smtlib

from xdsl_smt.pdl_constraints.integer_arith_constraints import (
    integer_arith_native_rewrites,
    integer_arith_native_constraints,
    integer_arith_native_static_constraints,
)


class OptMain(xDSLOptMain):
    def register_all_dialects(self):
        NEW_PDL = Dialect(
            "pdl",
            [*PDL.operations, *PDLDataflowDialect.operations],
            [*PDL.attributes, *PDLDataflowDialect.attributes],
        )
        self.ctx.register_dialect(Arith.name, lambda: Arith)
        self.ctx.register_dialect(Builtin.name, lambda: Builtin)
        self.ctx.register_dialect(Func.name, lambda: Func)
        self.ctx.register_dialect(Index.name, lambda: Index)
        self.ctx.register_dialect(SMTDialect.name, lambda: SMTDialect)
        self.ctx.register_dialect(SMTBitVectorDialect.name, lambda: SMTBitVectorDialect)
        self.ctx.register_dialect(SMTUtilsDialect.name, lambda: SMTUtilsDialect)
        self.ctx.register_dialect(SMTUBDialect.name, lambda: SMTUBDialect)
        self.ctx.register_dialect(Transfer.name, lambda: Transfer)
        self.ctx.register_dialect(Hoare.name, lambda: Hoare)
        self.ctx.register_dialect(PDL.name, lambda: NEW_PDL)
        self.ctx.register_dialect(Comb.name, lambda: Comb)
        self.ctx.register_dialect(HW.name, lambda: HW)
        self.ctx.register_dialect(LLVM.name, lambda: LLVM)

    def register_all_passes(self):
        super().register_all_passes()
        self.register_pass(LowerToSMT.name, lambda: LowerToSMT)
        self.register_pass(DeadCodeElimination.name, lambda: DeadCodeElimination)
        self.register_pass(CanonicalizeSMT.name, lambda: CanonicalizeSMT)
        self.register_pass(LowerPairs.name, lambda: LowerPairs)
        self.register_pass(PDLToSMT.name, lambda: PDLToSMT)

    def register_all_targets(self):
        super().register_all_targets()
        self.available_targets["smt"] = print_to_smtlib


def main():
    xdsl_main = OptMain()
    LowerToSMT.type_lowerers = [integer_poison_type_lowerer]
    LowerToSMT.attribute_semantics = {IntegerAttr: IntegerAttrSemantics()}
    LowerToSMT.operation_semantics = {**arith_semantics, **comb_semantics}

    PDLToSMT.native_rewrites = integer_arith_native_rewrites
    PDLToSMT.native_constraints = integer_arith_native_constraints
    PDLToSMT.native_static_constraints = integer_arith_native_static_constraints

    xdsl_main.run()


if __name__ == "__main__":
    main()
