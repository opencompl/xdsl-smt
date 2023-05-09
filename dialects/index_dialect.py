from xdsl.ir import Attribute, Dialect, OpResult
from xdsl.irdl import OpAttr, Operand, irdl_op_definition, IRDLOperation

from traits.effects import Pure


@irdl_op_definition
class Add(IRDLOperation, Pure):
    name = "index.add"
    lhs: Operand
    rhs: Operand
    result: OpResult

@irdl_op_definition
class And(IRDLOperation, Pure):
    name = "index.and"
    lhs: Operand

@irdl_op_definition
class Cmp(IRDLOperation, Pure):
    name = "index.cmp"
    lhs: Operand
    rhs: Operand
    predicate: OpAttr[Attribute]
    result: OpResult


@irdl_op_definition
class Constant(IRDLOperation, Pure):
    name = "index.constant"
    value: OpAttr[Attribute]
    result: OpResult

Index = Dialect(
    [
        Add,
        And,
        Cmp,
        Constant,
    ],
    [],
)
