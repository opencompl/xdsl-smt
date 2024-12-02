from dataclasses import dataclass
from typing import ClassVar
from xdsl.dialects.builtin import ModuleOp

from ..dialects import pdl_dataflow as pdl_dataflow

from xdsl.context import MLContext

from xdsl.passes import ModulePass
from xdsl_smt.passes.pdl_lowerers import PDLToSMTLowerer


@dataclass(frozen=True)
class PDLToSMT(ModulePass):
    name = "pdl-to-smt"

    pdl_lowerer: ClassVar[PDLToSMTLowerer] = PDLToSMTLowerer({}, {}, {})

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        self.pdl_lowerer.lower_to_smt(op, ctx)
