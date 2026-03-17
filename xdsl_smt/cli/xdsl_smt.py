#!/usr/bin/env python3
import argparse

from xdsl.ir import Dialect
from xdsl.xdsl_opt_main import xDSLOptMain

from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.dialects.pdl import PDL
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.dialects.test import Test
from xdsl.dialects.memref import MemRef
from xdsl_smt.dialects.smt_array_dialect import SMTArray
from xdsl_smt.dialects.smt_floatingpoint_dialect import SMTFloatingPointDialect
from xdsl_smt.dialects.smt_int_dialect import SMTIntDialect
from xdsl_smt.dialects.effects.effect import EffectDialect
from xdsl_smt.dialects.effects.ub_effect import UBEffectDialect
from xdsl_smt.dialects.effects.memory_effect import MemoryEffectDialect
from xdsl_smt.dialects.memory_dialect import MemoryDialect
from xdsl_smt.dialects.smt_tensor_dialect import SMTTensorDialect
from xdsl_smt.passes.lower_effects import LowerEffectPass
from xdsl_smt.passes.load_parametric_int_semantics import LoadIntSemanticsPass
from xdsl_smt.passes.lower_memory_effects import LowerMemoryEffectsPass
from xdsl_smt.passes.lower_effects_with_memory import LowerEffectsWithMemoryPass
from xdsl_smt.passes.lower_smt_tensor import LowerSMTTensor
from xdsl_smt.passes.merge_func_results import MergeFuncResultsPass
from xdsl_smt.passes.lower_memory_to_array import LowerMemoryToArrayPass
from xdsl_smt.passes.raise_llvm_to_func import RaiseLLVMToFunc
from xdsl_smt.passes.lower_abbv_to_bv import LowerAbbvToBvPass
from xdsl_smt.passes.resolve_transfer_widths import ResolveTransferWidths

from xdsl_smt.passes.dynamic_semantics import DynamicSemantics

from xdsl_smt.dialects.hoare_dialect import Hoare
from xdsl_smt.dialects.pdl_dataflow import PDLDataflowDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl_smt.dialects.index_dialect import Index
from xdsl_smt.dialects.transfer import Transfer
from xdsl_smt.dialects.hw_dialect import HW
from xdsl_smt.dialects.llvm_dialect import LLVM
from xdsl_smt.dialects.tv_dialect import TVDialect
from xdsl_smt.dialects.ub import UBDialect
from xdsl_smt.dialects.ab_bitvector_dialect import ABBitVectorDialect

from xdsl_smt.passes.dead_code_elimination import DeadCodeElimination
from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl_smt.passes.lower_to_smt import LowerToSMTPass
from xdsl_smt.passes.lower_ub_to_pairs import LowerUBToPairs
from xdsl_smt.passes.rewrite_smt_tensor import RewriteSMTTensor
from xdsl_smt.passes.smt_expand import SMTExpand
from xdsl_smt.passes.pdl_add_implicit_properties import PDLAddImplicitPropertiesPass

from xdsl_smt.passes.pdl_to_smt import PDLToSMT

from xdsl_smt.traits.smt_printer import print_to_smtlib

from xdsl_smt.pdl_constraints.integer_arith_constraints import (
    integer_arith_native_rewrites,
    integer_arith_native_constraints,
    integer_arith_native_static_constraints,
)
from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import (
    load_vanilla_semantics_with_transfer,
    load_vanilla_semantics_using_control_flow_dialects,
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
        self.ctx.register_dialect(SMTIntDialect.name, lambda: SMTIntDialect)
        self.ctx.register_dialect(SMTBitVectorDialect.name, lambda: SMTBitVectorDialect)
        self.ctx.register_dialect(SMTArray.name, lambda: SMTArray)
        self.ctx.register_dialect(SMTTensorDialect.name, lambda: SMTTensorDialect)
        self.ctx.register_dialect(SMTUtilsDialect.name, lambda: SMTUtilsDialect)
        self.ctx.register_dialect(
            SMTFloatingPointDialect.name, lambda: SMTFloatingPointDialect
        )
        self.ctx.register_dialect(EffectDialect.name, lambda: EffectDialect)
        self.ctx.register_dialect(UBEffectDialect.name, lambda: UBEffectDialect)
        self.ctx.register_dialect(MemoryEffectDialect.name, lambda: MemoryEffectDialect)
        self.ctx.register_dialect(MemoryDialect.name, lambda: MemoryDialect)
        self.ctx.register_dialect(Transfer.name, lambda: Transfer)
        self.ctx.register_dialect(TVDialect.name, lambda: TVDialect)
        self.ctx.register_dialect(Hoare.name, lambda: Hoare)
        self.ctx.register_dialect(PDL.name, lambda: NEW_PDL)
        self.ctx.register_dialect(Comb.name, lambda: Comb)
        self.ctx.register_dialect(HW.name, lambda: HW)
        self.ctx.register_dialect(LLVM.name, lambda: LLVM)
        self.ctx.register_dialect(Test.name, lambda: Test)
        self.ctx.register_dialect(MemRef.name, lambda: MemRef)
        self.ctx.register_dialect(UBDialect.name, lambda: UBDialect)
        self.ctx.register_dialect(ABBitVectorDialect.name, lambda: ABBitVectorDialect)
        self.ctx.load_registered_dialect(SMTDialect.name)
        self.ctx.load_registered_dialect(Transfer.name)
        self.ctx.load_registered_dialect(SMTIntDialect.name)
        self.ctx.load_registered_dialect(SMTBitVectorDialect.name)
        self.ctx.load_registered_dialect(SMTUtilsDialect.name)
        self.ctx.load_registered_dialect(SMTArray.name)
        self.ctx.load_registered_dialect(SMTFloatingPointDialect.name)
        self.ctx.load_registered_dialect(SMTTensorDialect.name)

    def register_all_passes(self):
        super().register_all_passes()
        self.register_pass(LowerToSMTPass.name, lambda: LowerToSMTPass)
        self.register_pass(LoadIntSemanticsPass.name, lambda: LoadIntSemanticsPass)
        self.register_pass(DeadCodeElimination.name, lambda: DeadCodeElimination)
        self.register_pass(LowerPairs.name, lambda: LowerPairs)
        self.register_pass(PDLToSMT.name, lambda: PDLToSMT)
        self.register_pass(LowerEffectPass.name, lambda: LowerEffectPass)
        self.register_pass(
            LowerEffectsWithMemoryPass.name, lambda: LowerEffectsWithMemoryPass
        )
        self.register_pass(LowerMemoryEffectsPass.name, lambda: LowerMemoryEffectsPass)
        self.register_pass(DynamicSemantics.name, lambda: DynamicSemantics)
        self.register_pass(MergeFuncResultsPass.name, lambda: MergeFuncResultsPass)
        self.register_pass(LowerMemoryToArrayPass.name, lambda: LowerMemoryToArrayPass)
        self.register_pass(LowerUBToPairs.name, lambda: LowerUBToPairs)
        self.register_pass(SMTExpand.name, lambda: SMTExpand)
        self.register_pass(
            PDLAddImplicitPropertiesPass.name, lambda: PDLAddImplicitPropertiesPass
        )
        self.register_pass(RaiseLLVMToFunc.name, lambda: RaiseLLVMToFunc)
        self.register_pass(LowerAbbvToBvPass.name, lambda: LowerAbbvToBvPass)
        self.register_pass(RewriteSMTTensor.name, lambda: RewriteSMTTensor)
        self.register_pass(LowerSMTTensor.name, lambda: LowerSMTTensor)
        self.register_pass(ResolveTransferWidths.name, lambda: ResolveTransferWidths)

    def register_all_targets(self):
        super().register_all_targets()
        self.available_targets["smt"] = print_to_smtlib

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        super().register_all_arguments(arg_parser)
        arg_parser.add_argument(
            "-s",
            "--semantics",
            type=str,
            choices=["default", "with-cf"],
            default="default",
            help="Choose the semantics to use for the transfer integers.",
        )


def main():
    xdsl_main = OptMain()
    if xdsl_main.args.semantics == "default":
        load_vanilla_semantics_with_transfer()
    else:
        load_vanilla_semantics_using_control_flow_dialects()

    PDLToSMT.pdl_lowerer.native_rewrites = integer_arith_native_rewrites
    PDLToSMT.pdl_lowerer.native_constraints = integer_arith_native_constraints
    PDLToSMT.pdl_lowerer.native_static_constraints = (
        integer_arith_native_static_constraints
    )
    xdsl_main.run()


if __name__ == "__main__":
    main()
