from xdsl.ir import Dialect, TypeAttribute, ParametrizedAttribute
from xdsl.irdl import (
    irdl_op_definition,
    irdl_attr_definition,
    IRDLOperation,
    operand_def,
    result_def,
)

from xdsl_smt.dialects.smt_dialect import BoolType


@irdl_attr_definition
class UBStateType(TypeAttribute, ParametrizedAttribute):
    """
    Type of a undefined behavior effect state.
    The undefined behavior effect follows the definition of LLVM and MLIR, where
    anything can happen in the semantics once undefined behavior is triggered.
    """

    name = "smt_ub.ub_state"


@irdl_op_definition
class CreateStateOp(IRDLOperation):
    """
    Create an empty UB state. In this state, no UB has been triggered yet.
    """

    name = "smt_ub.create_state"

    res = result_def(UBStateType)

    assembly_format = "attr-dict"


@irdl_op_definition
class TriggerOp(IRDLOperation):
    """Trigger undefined behavior."""

    name = "smt_ub.trigger"

    state = operand_def(UBStateType)
    res = result_def(UBStateType)

    assembly_format = "$state attr-dict"


@irdl_op_definition
class ToBoolOp(IRDLOperation):
    """
    Convert the current undefined behavior state to a boolean.
    Returns true if UB has been triggered, false otherwise.
    """

    name = "smt_ub.to_bool"

    state = operand_def(UBStateType)
    res = result_def(BoolType)

    assembly_format = "$state attr-dict"


SMTUBDialect = Dialect(
    "smt_ub",
    [
        CreateStateOp,
        TriggerOp,
        ToBoolOp,
    ],
    [UBStateType],
)
