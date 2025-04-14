"""
This file contains utilities to load a set of semantics.
It is used to load the default set of semantics we used in the project,
that is then used by the SMTLowerer.
"""

from xdsl.ir import Attribute, Operation
from xdsl.dialects import arith
from xdsl_smt.semantics.generic_integer_proxy import IntegerProxy
from xdsl_smt.semantics.arith_int_semantics import (
    IntConstantSemantics,
    get_binary_ef_semantics,
    IntSelectSemantics,
    IntCmpiSemantics,
    IntAndISemantics,
    get_div_semantics,
    IntIntegerTypeSemantics,
    IntIntegerAttrSemantics,
)
from xdsl_smt.semantics.semantics import OperationSemantics, AttributeSemantics
from xdsl_smt.dialects.transfer import (
    AbstractValueType,
    TransIntegerType,
)
from xdsl_smt.semantics.transfer_semantics import (
    AbstractValueTypeSemantics,
    TransferIntegerTypeSemantics,
)
from xdsl_smt.semantics.transfer_semantics import transfer_semantics
from xdsl.dialects.builtin import IntegerType, IntegerAttr, IndexType, MemRefType
from xdsl_smt.passes.lower_to_smt.smt_lowerer import SMTLowerer
from xdsl_smt.semantics.builtin_semantics import (
    IndexTypeSemantics,
    IntegerAttrSemantics,
    IntegerTypeSemantics,
)
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.comb_semantics import comb_semantics
from xdsl_smt.semantics.memref_semantics import memref_semantics, MemrefSemantics
from xdsl_smt.passes.lower_to_smt import (
    func_to_smt_patterns,
    transfer_to_smt_patterns,
)
from xdsl_smt.dialects import smt_int_dialect as smt_int


def load_vanilla_semantics_with_transfer(transfer_width: int):
    """Load our usual set of semantics with the transfer semantics."""
    load_vanilla_semantics()
    load_transfer_type_lowerer(transfer_width)


def load_vanilla_semantics():
    """Load our usual set of semantics."""
    SMTLowerer.type_lowerers = {
        IntegerType: IntegerTypeSemantics(),
        IndexType: IndexTypeSemantics(),
        MemRefType: MemrefSemantics(),
        AbstractValueType: AbstractValueTypeSemantics(),
    }
    SMTLowerer.attribute_semantics = {IntegerAttr: IntegerAttrSemantics()}
    SMTLowerer.op_semantics = {
        **arith_semantics,
        **comb_semantics,
        **memref_semantics,
        **transfer_semantics,
    }
    SMTLowerer.rewrite_patterns = {
        **func_to_smt_patterns,
        **transfer_to_smt_patterns,
    }


def load_transfer_type_lowerer(transfer_width: int):
    """Load the transfer type semantics."""
    SMTLowerer.type_lowerers = {
        **SMTLowerer.type_lowerers,
        **{TransIntegerType: TransferIntegerTypeSemantics(transfer_width)},
    }


def load_dynamic_semantics(semantics: dict[type[Operation], OperationSemantics]):
    """Load semantics defined in PDL."""
    SMTLowerer.dynamic_semantics_enabled = True
    SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}


def load_int_semantics(integer_proxy: IntegerProxy):
    """Load arith semantics that use the integer theory."""
    semantics = {
        arith.ConstantOp: IntConstantSemantics(integer_proxy),
        arith.SelectOp: IntSelectSemantics(integer_proxy),
        arith.AddiOp: get_binary_ef_semantics(smt_int.AddOp)(integer_proxy),
        arith.SubiOp: get_binary_ef_semantics(smt_int.SubOp)(integer_proxy),
        arith.MuliOp: get_binary_ef_semantics(smt_int.MulOp)(integer_proxy),
        arith.CmpiOp: IntCmpiSemantics(integer_proxy),
        arith.AndIOp: IntAndISemantics(integer_proxy),
        arith.DivUIOp: get_div_semantics(smt_int.DivOp)(integer_proxy),
        arith.RemUIOp: get_div_semantics(smt_int.ModOp)(integer_proxy),
    }
    SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}
    types = {
        TransIntegerType: IntIntegerTypeSemantics(integer_proxy),
        IntegerType: IntIntegerTypeSemantics(integer_proxy),
    }
    SMTLowerer.type_lowerers = {**SMTLowerer.type_lowerers, **types}
    attribute_semantics: dict[type[Attribute], AttributeSemantics] = {
        IntegerAttr: IntIntegerAttrSemantics()
    }
    SMTLowerer.attribute_semantics = {
        **SMTLowerer.attribute_semantics,
        **attribute_semantics,
    }
