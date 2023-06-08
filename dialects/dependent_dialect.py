"""A dialect that uses PDL to represent operations with depentent types."""

from typing import Annotated
from xdsl.ir import Dialect, OpResult, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import IRDLOperation, Operand, irdl_attr_definition, irdl_op_definition
from xdsl.dialects.pdl import TypeOp


@irdl_attr_definition
class DependentType(ParametrizedAttribute, TypeAttribute):
    name = "dep.type"


@irdl_op_definition
class DepArithAddiOp(IRDLOperation):
    """A dependent version of arith.addi"""

    name = "dep.arith.addi"

    lhs: Annotated[Operand, DependentType]
    rhs: Annotated[Operand, DependentType]

    type: Annotated[Operand, TypeOp]
    """The type of all operands and results"""

    result: Annotated[OpResult, DependentType]


Dep = Dialect([DepArithAddiOp], [DependentType])
