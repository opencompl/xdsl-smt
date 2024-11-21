from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp

from xdsl_smt.semantics.accessor import PowEnabledIntAccessor as IntAccessor
from xdsl_smt.semantics.load_int_semantics import (
    load_int_semantics,
)


@dataclass(frozen=True)
class LoadIntSemanticsPass(ModulePass):
    name = "load-int-semantics"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        accessor = IntAccessor()
        load_int_semantics(accessor)
