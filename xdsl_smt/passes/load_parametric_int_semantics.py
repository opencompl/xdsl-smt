from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerType
from xdsl_smt.passes.lower_to_smt.lower_to_smt import SMTLowerer
from xdsl_smt.semantics.arith_int_semantics import (
    IntIntegerTypeSemantics,
    IntConstantSemantics,
    IntAddiSemantics,
    IntSubiSemantics,
    IntMuliSemantics,
    IntCmpiSemantics,
    IntDivUISemantics,
    IntRemUISemantics,
)


@dataclass(frozen=True)
class LoadIntSemanticsPass(ModulePass):
    name = "load-int-semantics"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        semantics = {
            arith.Constant: IntConstantSemantics(),
            arith.Addi: IntAddiSemantics(),
            arith.Subi: IntSubiSemantics(),
            arith.Muli: IntMuliSemantics(),
            arith.Cmpi: IntCmpiSemantics(),
            arith.DivUI: IntDivUISemantics(),
            arith.RemUI: IntRemUISemantics(),
        }
        SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}
        types = {
            IntegerType: IntIntegerTypeSemantics(),
        }
        SMTLowerer.type_lowerers = {**SMTLowerer.type_lowerers, **types}
