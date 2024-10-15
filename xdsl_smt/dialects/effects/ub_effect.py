from xdsl.ir import Dialect, SSAValue
from xdsl.irdl import (
    irdl_op_definition,
    IRDLOperation,
    operand_def,
    result_def,
)

from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.dialects.effects.effect import StateType
from xdsl.traits import Pure


@irdl_op_definition
class CreateStateOp(IRDLOperation):
    """
    Create an empty UB state. In this state, no UB has been triggered yet.
    """

    name = "ub_effect.create_state"

    res = result_def(StateType())

    assembly_format = "attr-dict"

    traits = frozenset([Pure()])

    def __init__(self):
        super().__init__(result_types=[StateType()])


@irdl_op_definition
class TriggerOp(IRDLOperation):
    """Trigger undefined behavior."""

    name = "ub_effect.trigger"

    state = operand_def(StateType())
    res = result_def(StateType())

    assembly_format = "$state attr-dict"

    traits = frozenset([Pure()])

    def __init__(self, state: SSAValue):
        super().__init__(operands=[state], result_types=[StateType()])


@irdl_op_definition
class ToBoolOp(IRDLOperation):
    """
    Convert the current undefined behavior state to a boolean.
    Returns true if UB has been triggered, false otherwise.
    """

    name = "ub_effect.to_bool"

    state = operand_def(StateType())
    res = result_def(BoolType())

    assembly_format = "$state attr-dict"

    traits = frozenset([Pure()])

    def __init__(self, state: SSAValue):
        super().__init__(operands=[state], result_types=[BoolType()])


UBEffectDialect = Dialect(
    "ub_effect",
    [
        CreateStateOp,
        TriggerOp,
        ToBoolOp,
    ],
    [],
)
