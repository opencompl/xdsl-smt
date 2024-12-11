from xdsl.ir import Operation
from xdsl_smt.semantics.semantics import OperationSemantics
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


def load_vanilla_semantics():
    SMTLowerer.type_lowerers = {
        IntegerType: IntegerTypeSemantics(),
        IndexType: IndexTypeSemantics(),
        MemRefType: MemrefSemantics(),
    }
    SMTLowerer.attribute_semantics = {IntegerAttr: IntegerAttrSemantics()}
    SMTLowerer.op_semantics = {**arith_semantics, **comb_semantics, **memref_semantics}
    SMTLowerer.rewrite_patterns = {**func_to_smt_patterns, **transfer_to_smt_patterns}


def load_dynamic_semantics(semantics: dict[type[Operation], OperationSemantics]):
    SMTLowerer.dynamic_semantics_enabled = True
    SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}


def load_transfer_semantics(width: int):
    SMTLowerer.rewrite_patterns = {
        **func_to_smt_patterns,
    }
    SMTLowerer.type_lowerers = {
        IntegerType: IntegerTypeSemantics(),
        AbstractValueType: AbstractValueTypeSemantics(),
        TransIntegerType: TransferIntegerTypeSemantics(width),
    }
    SMTLowerer.op_semantics = {
        **arith_semantics,
        **transfer_semantics,
        **comb_semantics,
    }
