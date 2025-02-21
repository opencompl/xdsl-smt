import abc
from xdsl.irdl import IRDLOperation, VarConstraint
from xdsl.pattern_rewriter import (
    PatternRewriter,
)
from xdsl.ir import Operation

from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import transfer
from xdsl_smt.dialects.smt_tensor_dialect import (
    SMTTensorType,
    TensorAddOp,
    TensorMultiplyOp,
    TensorTransposeOp,
    TensorSubtractOp,
)
from xdsl_smt.passes.lower_to_smt.smt_lowerer import (
    SMTLowerer,
)
from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.semantics.semantics import OperationSemantics, TypeSemantics
from xdsl.ir import Operation, SSAValue, Attribute
from typing import Mapping, Sequence, ClassVar
from xdsl.utils.isattr import isattr
from xdsl.dialects.builtin import IntegerAttr, IntegerType, TensorType
import xdsl.dialects.stablehlo as stablehlo


class TensorTypeSemantics(TypeSemantics):
    """Lower all tensor types in stable HLO to SMT tensor types
    But the last element is useless, this makes GetOp easier"""

    def get_semantics(self, type: Attribute) -> Attribute:
        if not isinstance(type, TensorType):
            raise ValueError("Expect a tensor type")
        elementType = SMTLowerer.lower_type(type.element_type)
        tensorType = SMTTensorType(elementType, type.shape, type.encoding)
        return tensorType


class StableHLOAddOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert isinstance(operands[0].type, SMTTensorType)
        tensorAddOp = TensorAddOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(tensorAddOp)
        return ((tensorAddOp.result,), effect_state)


class StableHLOSubtractOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert isinstance(operands[0].type, SMTTensorType)
        tensorAddOp = TensorSubtractOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(tensorAddOp)
        return ((tensorAddOp.result,), effect_state)


class StableHLOMultiplyOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert isinstance(operands[0].type, SMTTensorType)
        tensorAddOp = TensorMultiplyOp(operands[0], operands[1])
        rewriter.insert_op_before_matched_op(tensorAddOp)
        return ((tensorAddOp.result,), effect_state)


class StableHLOTransposeOpSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        assert isinstance(operands[0].type, SMTTensorType)
        if not isinstance(results[0], TensorType):
            raise ValueError("Expect a tensor type")
        elementType = SMTLowerer.lower_type(results[0].element_type)
        tensorType = SMTTensorType(elementType, results[0].shape, results[0].encoding)
        tensorTransposeOp = TensorTransposeOp(
            operands[0], attributes["permutation"], tensorType
        )
        rewriter.insert_op_before_matched_op(tensorTransposeOp)
        return ((tensorTransposeOp.result,), effect_state)


stablehlo_semantics: dict[type[Operation], OperationSemantics] = {
    stablehlo.AddOp: StableHLOAddOpSemantics(),
    stablehlo.MultiplyOp: StableHLOMultiplyOpSemantics(),
    stablehlo.TransposeOp: StableHLOTransposeOpSemantics(),
    stablehlo.SubtractOp: StableHLOSubtractOpSemantics(),
}
