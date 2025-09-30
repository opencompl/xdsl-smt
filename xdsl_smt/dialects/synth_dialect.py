from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import (
    irdl_op_definition,
    IRDLOperation,
    result_def,
    operand_def,
    traits_def,
)
from xdsl.traits import Pure


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "synth.constant"

    res = result_def()

    def __init__(self, type: Attribute):
        super().__init__(result_types=[type])

    assembly_format = "attr-dict `:` type($res)"


@irdl_op_definition
class FromValueOp(IRDLOperation):
    name = "synth.from_value"

    input = operand_def()

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input])

    assembly_format = "attr-dict $input `:` type($input)"


@irdl_op_definition
class ConversionOp(IRDLOperation):
    name = "synth.conversion"

    input = operand_def()
    output = result_def()

    traits = traits_def(Pure())

    def __init__(self, input: SSAValue, output_type: Attribute):
        super().__init__(operands=[input], result_types=[output_type])

    assembly_format = "attr-dict $input `:` type($input) `->` type($output)"


SynthDialect = Dialect("synth", [ConstantOp, FromValueOp, ConversionOp])
