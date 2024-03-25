from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Dialect
from xdsl.irdl import (
    VarOpResult,
    VarOperand,
    operand_def,
    IRDLOperation,
    Operand,
    irdl_op_definition,
    var_operand_def,
    var_result_def,
    attr_def,
)
from xdsl.dialects.pdl import ValueType


@irdl_op_definition
class GetOp(IRDLOperation):
    """Get the dataflow analysis of an operand."""

    name = "pdl.dataflow.get"

    value: Operand = operand_def(ValueType)
    res: VarOpResult = var_result_def(Attribute)

    domain_name: StringAttr = attr_def(StringAttr)


@irdl_op_definition
class AttachOp(IRDLOperation):
    """Attach an analysis to an operation."""

    name = "pdl.dataflow.attach"

    value: Operand = operand_def(ValueType)
    domains: VarOperand = var_operand_def(Attribute)
    domain_name: StringAttr = attr_def(StringAttr)


PDLDataflowDialect = Dialect("pdl", [GetOp, AttachOp])
