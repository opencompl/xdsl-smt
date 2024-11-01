from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects import arith
from xdsl_smt.passes.lower_to_smt.lower_to_smt import SMTLowerer
from xdsl_smt.semantics.arith_int_semantics import (
    IntConstantSemantics,
    IntAddSemantics,
    IntSubSemantics,
    IntMulSemantics,
)


@dataclass(frozen=True)
class LoadParametricIntSemantics(ModulePass):
    name = "load-int-semantics"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        semantics = {
            arith.Constant: IntConstantSemantics(),
            arith.Addi: IntAddSemantics(),
            arith.Subi: IntSubSemantics(),
            arith.Muli: IntMulSemantics(),
        }
        SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}