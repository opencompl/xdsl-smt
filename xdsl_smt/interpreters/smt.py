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

    @impl(smt.AndOp)
    def run_and(
        self, interpreter: Interpreter, op: smt.AndOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(args[0], bool)
        assert isinstance(args[1], bool)
        return (args[0] and args[1],)

    @impl(smt.OrOp)
    def run_or(
        self, interpreter: Interpreter, op: smt.OrOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(args[0], bool)
        assert isinstance(args[1], bool)
        return (args[0] or args[1],)

    @impl(smt.ImpliesOp)
    def run_implies(
        self, interpreter: Interpreter, op: smt.ImpliesOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(args[0], bool)
        assert isinstance(args[1], bool)
        return (not args[0] or args[1],)

    @impl(smt.XorOp)
    def run_xor(
        self, interpreter: Interpreter, op: smt.XorOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(args[0], bool)
        assert isinstance(args[1], bool)
        return (args[0] != args[1],)

    @impl(smt.DistinctOp)
    def run_distinct(
        self, interpreter: Interpreter, op: smt.DistinctOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return (args[0] != args[1],)

    @impl(smt.EqOp)
    def run_eq(
        self, interpreter: Interpreter, op: smt.EqOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return (args[0] == args[1],)

    @impl(smt.IteOp)
    def run_ite(
        self, interpreter: Interpreter, op: smt.IteOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(args[0], bool)
        return (args[1] if args[0] else args[2],)
