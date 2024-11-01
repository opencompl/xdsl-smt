#!/usr/bin/env python3

from xdsl.ir import Dialect
from xdsl.xdsl_opt_main import xDSLOptMain

from xdsl.dialects.builtin import Builtin, IntegerAttr, IntegerType, IndexType
from xdsl.dialects.func import Func
from xdsl.dialects.pdl import PDL
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.dialects.test import Test
from xdsl.dialects.memref import MemRef
from xdsl_smt.dialects.effects.effect import EffectDialect
from xdsl_smt.dialects.effects.ub_effect import UBEffectDialect
from xdsl_smt.dialects.effects.memory_effect import MemoryEffectDialect
from xdsl_smt.passes.lower_effects import LowerEffectPass
from xdsl_smt.passes.lower_to_smt.lower_to_smt import SMTLowerer
from xdsl_smt.semantics.memref_semantics import memref_semantics
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.builtin_semantics import (
    IndexTypeSemantics,
    IntegerAttrSemantics,
    IntegerTypeSemantics,
)

from xdsl_smt.passes.lower_to_smt import (
    func_to_smt_patterns,
    transfer_to_smt_patterns,
)

from xdsl_smt.passes.dynamic_semantics import DynamicSemantics
from xdsl_smt.passes.load_parametric_int_semantics import LoadParametricIntSemantics

from xdsl_smt.dialects.hoare_dialect import Hoare
from xdsl_smt.dialects.pdl_dataflow import PDLDataflowDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_int_dialect import SMTIntDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl_smt.dialects.index_dialect import Index
from xdsl_smt.dialects.transfer import Transfer
from xdsl_smt.dialects.hw_dialect import HW
from xdsl_smt.dialects.llvm_dialect import LLVM
from xdsl_smt.dialects.tv_dialect import TVDialect

from xdsl_smt.passes.dead_code_elimination import DeadCodeElimination
from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl_smt.passes.lower_to_smt import LowerToSMTPass
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
        self.ctx.register_dialect(SMTIntDialect.name, lambda: SMTIntDialect)
        self.ctx.register_dialect(SMTUtilsDialect.name, lambda: SMTUtilsDialect)
        self.ctx.register_dialect(EffectDialect.name, lambda: EffectDialect)
        self.ctx.register_dialect(UBEffectDialect.name, lambda: UBEffectDialect)
        self.ctx.register_dialect(MemoryEffectDialect.name, lambda: MemoryEffectDialect)
        self.ctx.register_dialect(Transfer.name, lambda: Transfer)
        self.ctx.register_dialect(TVDialect.name, lambda: TVDialect)
        self.ctx.register_dialect(Hoare.name, lambda: Hoare)
        self.ctx.register_dialect(PDL.name, lambda: NEW_PDL)
        self.ctx.register_dialect(Comb.name, lambda: Comb)
        self.ctx.register_dialect(HW.name, lambda: HW)
        self.ctx.register_dialect(LLVM.name, lambda: LLVM)
        self.ctx.register_dialect(Test.name, lambda: Test)
        self.ctx.register_dialect(MemRef.name, lambda: MemRef)
        self.ctx.load_registered_dialect(SMTDialect.name)
        self.ctx.load_registered_dialect(SMTBitVectorDialect.name)
        self.ctx.load_registered_dialect(SMTIntDialect.name)
        self.ctx.load_registered_dialect(SMTUtilsDialect.name)

    def register_all_passes(self):
        super().register_all_passes()
        self.register_pass(LowerToSMTPass.name, lambda: LowerToSMTPass)
        self.register_pass(DeadCodeElimination.name, lambda: DeadCodeElimination)
        self.register_pass(LowerPairs.name, lambda: LowerPairs)
        self.register_pass(PDLToSMT.name, lambda: PDLToSMT)
        self.register_pass(LowerEffectPass.name, lambda: LowerEffectPass)
        self.register_pass(DynamicSemantics.name, lambda: DynamicSemantics)
        self.register_pass(
            LoadParametricIntSemantics.name, lambda: LoadParametricIntSemantics
        )

    def register_all_targets(self):
        super().register_all_targets()
        self.available_targets["smt"] = print_to_smtlib


def main():
    xdsl_main = OptMain()
    SMTLowerer.type_lowerers = {
        IntegerType: IntegerTypeSemantics(),
        IndexType: IndexTypeSemantics(),
    }
    SMTLowerer.attribute_semantics = {IntegerAttr: IntegerAttrSemantics()}
    SMTLowerer.op_semantics = {**arith_semantics, **comb_semantics, **memref_semantics}
    SMTLowerer.rewrite_patterns = {**func_to_smt_patterns, **transfer_to_smt_patterns}

    PDLToSMT.pdl_lowerer.native_rewrites = integer_arith_native_rewrites
    PDLToSMT.pdl_lowerer.native_constraints = integer_arith_native_constraints
    PDLToSMT.pdl_lowerer.native_static_constraints = (
        integer_arith_native_static_constraints
    )

    xdsl_main.run()


if __name__ == "__main__":
    main()
