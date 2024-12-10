from xdsl_smt.semantics.accessor import IntAccessor
from xdsl.ir import Attribute
from xdsl_smt.semantics.semantics import AttributeSemantics
from xdsl.dialects import arith
from xdsl_smt.semantics.arith_int_semantics import (
    IntConstantSemantics,
    get_binary_ef_semantics,
    IntSelectSemantics,
    IntCmpiSemantics,
    IntExtUISemantics,
    IntExtSISemantics,
    IntTruncISemantics,
    IntAndISemantics,
    get_div_semantics,
    IntIntegerTypeSemantics,
    IntIntegerAttrSemantics,
)
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.dialects import transfer
from xdsl.dialects.builtin import IntegerType, IntegerAttr, IndexType
from xdsl_smt.passes.pdl_to_smt_context import PDLToSMTRewriteContext
from xdsl_smt.passes.lowerers import SMTLowerer
from xdsl_smt.semantics.builtin_semantics import (
    IndexTypeSemantics,
    IntegerAttrSemantics,
    IntegerTypeSemantics,
)
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.comb_semantics import comb_semantics
from xdsl_smt.semantics.memref_semantics import memref_semantics
from xdsl_smt.passes.lower_to_smt import (
    func_to_smt_patterns,
    transfer_to_smt_patterns,
)


def load_vanilla_semantics():
    SMTLowerer.type_lowerers = {
        IntegerType: IntegerTypeSemantics(),
        IndexType: IndexTypeSemantics(),
    }
    SMTLowerer.attribute_semantics = {IntegerAttr: IntegerAttrSemantics()}
    SMTLowerer.op_semantics = {**arith_semantics, **comb_semantics, **memref_semantics}
    SMTLowerer.rewrite_patterns = {**func_to_smt_patterns, **transfer_to_smt_patterns}


def load_int_semantics(accessor: IntAccessor):
    semantics = {
        arith.Constant: IntConstantSemantics(accessor),
        arith.Select: IntSelectSemantics(accessor),
        arith.Addi: get_binary_ef_semantics(smt_int.AddOp)(accessor),
        arith.Subi: get_binary_ef_semantics(smt_int.SubOp)(accessor),
        arith.Muli: get_binary_ef_semantics(smt_int.MulOp)(accessor),
        arith.Cmpi: IntCmpiSemantics(accessor),
        arith.AndI: IntAndISemantics(accessor),
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
        arith.ExtSIOp: IntExtSISemantics(accessor, context),
        arith.TruncIOp: IntTruncISemantics(accessor, context),
    }
    SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}
