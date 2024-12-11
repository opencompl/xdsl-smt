from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerType
from xdsl_smt.passes.lower_to_smt.smt_lowerer import SMTLowerer
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.semantics.arith_int_semantics import (
    IntIntegerTypeSemantics,
    IntConstantSemantics,
    IntCmpiSemantics,
    get_binary_ef_semantics,
    get_div_semantics,
)


@dataclass(frozen=True)
class LoadIntSemanticsPass(ModulePass):
    name = "load-int-semantics"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        semantics = {
            arith.Constant: IntConstantSemantics(),
            arith.Addi: get_binary_ef_semantics(smt_int.AddOp)(),
            arith.Subi: get_binary_ef_semantics(smt_int.SubOp)(),
            arith.Muli: get_binary_ef_semantics(smt_int.MulOp)(),
            arith.Cmpi: IntCmpiSemantics(),
            arith.DivUI: get_div_semantics(smt_int.DivOp)(),
            arith.RemUI: get_div_semantics(smt_int.ModOp)(),
        }
        SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}
        types = {
            IntegerType: IntIntegerTypeSemantics(),
        }
        SMTLowerer.type_lowerers = {**SMTLowerer.type_lowerers, **types}
