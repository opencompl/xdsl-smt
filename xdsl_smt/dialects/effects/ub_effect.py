from typing import Sequence
from xdsl.ir import Dialect, SSAValue, Attribute
from xdsl.irdl import (
    irdl_op_definition,
    IRDLOperation,
    operand_def,
    result_def,
    var_result_def,
    traits_def,
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

    traits = traits_def(Pure())

    def __init__(self):
        super().__init__(result_types=[StateType()])


@irdl_op_definition
class TriggerOp(IRDLOperation):
    """Trigger undefined behavior."""

    name = "ub_effect.trigger"

    state = operand_def(StateType())
    res = result_def(StateType())
    irrelevant_values = var_result_def()

    assembly_format = "$state attr-dict (`:` type($irrelevant_values)^)?"

    traits = traits_def(Pure())

    def __init__(self, state: SSAValue, irrelevant_values: Sequence[Attribute] = ()):
        super().__init__(
            operands=[state], result_types=[StateType(), irrelevant_values]
        )


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

    traits = traits_def(Pure())

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
