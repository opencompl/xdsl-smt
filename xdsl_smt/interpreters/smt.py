from typing import Any, cast

from xdsl.ir import Attribute

from xdsl_smt.dialects import smt_dialect as smt
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    TerminatorValue,
    impl_attr,
    impl,
    impl_terminator,
    register_impls,
)


@register_impls
class SMTFunctions(InterpreterFunctions):
    @impl_attr(smt.BoolType)
    def bool_attr_value(
        self, interpreter: Interpreter, attr: Attribute, attr_type: smt.BoolType
    ) -> float:
        interpreter.interpreter_assert(isinstance(attr, smt.BoolAttr))
        attr = cast(smt.BoolAttr, attr)
        return attr.data

    @impl_terminator(smt.ReturnOp)
    def run_return(
        self, interpreter: Interpreter, op: smt.ReturnOp, args: tuple[Any, ...]
    ) -> tuple[TerminatorValue, PythonValues]:
        return ReturnedValues(args), ()

    @impl(smt.NotOp)
    def run_not(
        self, interpreter: Interpreter, op: smt.NotOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(args[0], bool)
        return (not args[0],)
