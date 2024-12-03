from dataclasses import dataclass
from typing import ClassVar
from xdsl.dialects.builtin import ModuleOp

from ..dialects import pdl_dataflow as pdl_dataflow

from xdsl.context import MLContext

from xdsl.passes import ModulePass
from xdsl_smt.passes.pdl_lowerers import IntPDLToSMTLowerer, BVPDLToSMTLowerer

from xdsl_smt.semantics.arith_int_semantics import (
    trigger_parametric_int,
)


@dataclass(frozen=True)
class PDLToSMT(ModulePass):
    name = "pdl-to-smt"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        if trigger_parametric_int(op):
            lowerer = IntPDLToSMTLowerer({}, {}, {})
        else:
            lowerer = BVPDLToSMTLowerer({}, {}, {})

        lowerer.lower_to_smt(op, ctx)
