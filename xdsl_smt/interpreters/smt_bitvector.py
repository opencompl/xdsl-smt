from typing import Any, cast

from xdsl.ir import Attribute

from xdsl.dialects.smt import BitVectorType
from xdsl.dialects.builtin import IntegerAttr
from xdsl_smt.dialects import smt_bitvector_dialect as bv
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl_attr,
    impl,
    register_impls,
)


@register_impls
class SMTBitVectorFunctions(InterpreterFunctions):
    @impl_attr(BitVectorType)
    def bv_attr_value(
        self, interpreter: Interpreter, attr: Attribute, attr_type: BitVectorType
    ) -> int:
        interpreter.interpreter_assert(isinstance(attr, IntegerAttr))
        attr = cast(IntegerAttr, attr)
        return attr.value.data

    @impl(bv.ConstantOp)
    def run_bv_constant(
        self, interpreter: Interpreter, op: bv.ConstantOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return (op.value,)
