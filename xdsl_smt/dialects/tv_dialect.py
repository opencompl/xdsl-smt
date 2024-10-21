from typing import Sequence
from xdsl.ir import Dialect, SSAValue
from xdsl.irdl import (
    irdl_op_definition,
    IRDLOperation,
    operand_def,
    var_operand_def,
    result_def,
)
from xdsl.irdl import SameVariadicOperandSize

from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl_smt.dialects.effects.effect import StateType


@irdl_op_definition
class EffectfulRefinementOp(IRDLOperation):
    """
    Check if a pair of effect states and values are in a refinement relation.
    """

    name = "effect.effectful_refinement"

    state_before = operand_def(StateType())
    value_before = var_operand_def()
    state_after = operand_def(StateType())
    value_after = var_operand_def()

    res = result_def(BoolType())

    assembly_format = "`(` $state_before `,` $value_before `)` `to` `(` $state_after `,` $value_after `)` attr-dict `:` type($value_before) `to` type($value_after)"

    irdl_options = [SameVariadicOperandSize()]

    def __init__(
        self,
        state_before: SSAValue,
        value_before: Sequence[SSAValue],
        state_after: SSAValue,
        value_after: Sequence[SSAValue],
    ):
        super().__init__(
            operands=[state_before, value_before, state_after, value_after],
            result_types=[BoolType()],
        )


@irdl_op_definition
class RefinementOp(IRDLOperation):
    """
    Represent a refinement between two values.
    """

    name = "tv.refinement"

    value_before = operand_def()
    value_after = operand_def()

    res = result_def(BoolType())

    assembly_format = "$value_before `to` $value_after attr-dict `:` type($value_before) `to` type($value_after)"

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
