from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp, IntegerAttr
from xdsl.dialects import arith
from xdsl.dialects.builtin import IntegerType
from xdsl_smt.passes.lower_to_smt.lower_to_smt import SMTLowerer
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.semantics.arith_int_semantics import (
    FixedWidthIntAccessor,
    IntIntegerTypeSemantics,
    IntIntegerAttrSemantics,
    IntConstantSemantics,
    IntCmpiSemantics,
    get_binary_ef_semantics,
    get_div_semantics,
)


@dataclass(frozen=True)
class LoadIntSemanticsPass(ModulePass):
    name = "load-int-semantics"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        accessor = FixedWidthIntAccessor()
        # Override SMT semantics
        semantics = {
            arith.Constant: IntConstantSemantics(accessor),
            arith.Addi: get_binary_ef_semantics(smt_int.AddOp)(accessor),
            arith.Subi: get_binary_ef_semantics(smt_int.SubOp)(accessor),
            arith.Muli: get_binary_ef_semantics(smt_int.MulOp)(accessor),
            arith.Cmpi: IntCmpiSemantics(accessor),
            arith.DivUI: get_div_semantics(smt_int.DivOp)(accessor),
            arith.RemUI: get_div_semantics(smt_int.ModOp)(accessor),
        }
        SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}
        types = {
            IntegerType: IntIntegerTypeSemantics(accessor),
        }
        SMTLowerer.type_lowerers = {**SMTLowerer.type_lowerers, **types}
        attribute_semantics = {IntegerAttr: IntIntegerAttrSemantics()}
        SMTLowerer.attribute_semantics = {
            **SMTLowerer.attribute_semantics,
            **attribute_semantics,
        }
