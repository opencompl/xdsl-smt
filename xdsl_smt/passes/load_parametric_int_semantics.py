from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl_smt.semantics.generic_integer_proxy import IntegerProxy
from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import load_int_semantics


@dataclass(frozen=True)
class LoadIntSemanticsPass(ModulePass):
    name = "load-int-semantics"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        integer_proxy = IntegerProxy()
        load_int_semantics(integer_proxy)
