from xdsl.ir import Operation
from xdsl_smt.dialects.smt_tensor_dialect import SMTTensorType
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
from xdsl.dialects.builtin import (
    IntegerType,
    IntegerAttr,
    IndexType,
    MemRefType,
    TensorType,
)
from xdsl_smt.passes.lower_to_smt.smt_lowerer import SMTLowerer
from xdsl_smt.semantics.builtin_semantics import (
    IndexTypeSemantics,
    IntegerAttrSemantics,
    IntegerTypeSemantics,
)
from xdsl_smt.semantics.stablehlo_semantics import (
    TensorTypeSemantics,
    stablehlo_semantics,
)
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.comb_semantics import comb_semantics
from xdsl_smt.semantics.memref_semantics import memref_semantics, MemrefSemantics
from xdsl_smt.passes.lower_to_smt import (
    func_to_smt_patterns,
    transfer_to_smt_patterns,
)


def load_vanilla_semantics_with_transfer(transfer_width: int):
    load_vanilla_semantics()
    load_transfer_type_lowerer(transfer_width)


def load_vanilla_semantics():
    SMTLowerer.type_lowerers = {
        IntegerType: IntegerTypeSemantics(),
        IndexType: IndexTypeSemantics(),
        MemRefType: MemrefSemantics(),
        AbstractValueType: AbstractValueTypeSemantics(),
        TensorType: TensorTypeSemantics(),
    }
    SMTLowerer.attribute_semantics = {IntegerAttr: IntegerAttrSemantics()}
    SMTLowerer.op_semantics = {
        **arith_semantics,
        **comb_semantics,
        **memref_semantics,
        **transfer_semantics,
        **stablehlo_semantics,
    }
    SMTLowerer.rewrite_patterns = {
        **func_to_smt_patterns,
        **transfer_to_smt_patterns,
    }


def load_transfer_type_lowerer(transfer_width: int):
    SMTLowerer.type_lowerers = {
        **SMTLowerer.type_lowerers,
        **{TransIntegerType: TransferIntegerTypeSemantics(transfer_width)},
    }


def load_dynamic_semantics(semantics: dict[type[Operation], OperationSemantics]):
    SMTLowerer.dynamic_semantics_enabled = True
    SMTLowerer.op_semantics = {**SMTLowerer.op_semantics, **semantics}
