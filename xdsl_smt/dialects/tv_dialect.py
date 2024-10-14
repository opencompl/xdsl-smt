from typing import Annotated
from xdsl.ir import Dialect, Attribute, SSAValue
from xdsl.irdl import (
    irdl_op_definition,
    IRDLOperation,
    operand_def,
    ConstraintVar,
    result_def,
)

from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.dialects.effects.effect import StateType


@irdl_op_definition
class EffectfulRefinementOp(IRDLOperation):
    """
    Check if a pair of effect states and values are in a refinement relation.
    """

    name = "effect.refinement"

    T = Annotated[Attribute, ConstraintVar("T")]

    state_before = operand_def(StateType())
    value_before = operand_def(T)
    state_after = operand_def(StateType())
    value_after = operand_def(T)

    res = result_def(BoolType())

    assembly_format = "`(` $state_before `,` $value_before `)` `to` `(` $state_after `,` $value_after `)` `:` type($value_before) attr-dict"

    def __init__(
        self,
        state_before: SSAValue,
        value_before: SSAValue,
        state_after: SSAValue,
        value_after: SSAValue,
    ):
        super().__init__(
            operands=[state_before, value_before, state_after, value_after],
            result_types=[BoolType()],
        )


@irdl_op_definition
class RefinementOp(IRDLOperation):
    """
    Represent a refinement operation.
    The refinement semantics is defined
    """

    name = "tv.effectful_refinement"

    T = Annotated[Attribute, ConstraintVar("T")]

    value_before = operand_def(T)
    value_after = operand_def(T)

    res = result_def(BoolType())

    def __init__(self, value_before: SSAValue, value_after: SSAValue):
        super().__init__(
            operands=[value_before, value_after], result_types=[BoolType()]
        )


TVDialect = Dialect(
    "tv",
    [
        EffectfulRefinementOp,
        RefinementOp,
    ],
    [],
)
