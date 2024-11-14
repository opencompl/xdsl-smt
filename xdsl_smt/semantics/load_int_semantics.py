from xdsl_smt.semantics.accessor import IntAccessor
from xdsl.dialects import arith, pdl, comb
from xdsl_smt.semantics.arith_int_semantics import (
    IntConstantSemantics,
    get_binary_ef_semantics,
    IntCmpiSemantics,
    IntExtUISemantics,
    IntAndISemantics,
    IntOrISemantics,
    IntXOrISemantics,
    get_div_semantics,
    IntIntegerTypeSemantics,
    IntIntegerAttrSemantics,
)
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.passes.lower_to_smt.lower_to_smt import SMTLowerer
from xdsl_smt.dialects import transfer
from xdsl.dialects.builtin import IntegerType, AnyIntegerAttr, IntegerAttr, FunctionType
from xdsl_smt.passes.pdl_to_smt_context import PDLToSMTRewriteContext


def load_int_semantics(accessor: IntAccessor):
    semantics = {
        arith.Constant: IntConstantSemantics(accessor),
        arith.Addi: get_binary_ef_semantics(smt_int.AddOp)(accessor),
        arith.Subi: get_binary_ef_semantics(smt_int.SubOp)(accessor),
        arith.Muli: get_binary_ef_semantics(smt_int.MulOp)(accessor),
        arith.Cmpi: IntCmpiSemantics(accessor),
        arith.AndI: IntAndISemantics(accessor),
        arith.OrI: IntOrISemantics(accessor),
        arith.XOrI: IntXOrISemantics(accessor),
        arith.DivUI: get_div_semantics(smt_int.DivOp)(accessor),
        arith.RemUI: get_div_semantics(smt_int.ModOp)(accessor),
    }
    SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}
    types = {
        transfer.TransIntegerType: IntIntegerTypeSemantics(accessor),
        IntegerType: IntIntegerTypeSemantics(accessor),
    }
    SMTLowerer.type_lowerers = {**SMTLowerer.type_lowerers, **types}
    attribute_semantics: dict[type[Attribute], AttributeSemantics] = {
        IntegerAttr: IntIntegerAttrSemantics()
    }
    SMTLowerer.attribute_semantics = {
        **SMTLowerer.attribute_semantics,
        **attribute_semantics,
    }


def load_int_semantics_with_context(
    accessor: IntAccessor, context: PDLToSMTRewriteContext
):
    load_int_semantics(accessor)
    semantics = {
        arith.ExtUIOp: IntExtUISemantics(accessor, context),
    }
    SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}
