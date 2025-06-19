from typing import Any, cast

from xdsl.ir import Attribute

from xdsl.dialects.builtin import IntegerAttr
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
    ) -> bool:
        interpreter.interpreter_assert(isinstance(attr, IntegerAttr))
        attr = cast(IntegerAttr, attr)
        return attr.value.data != 0

    @impl_terminator(smt.ReturnOp)
    def run_return(
        self, interpreter: Interpreter, op: smt.ReturnOp, args: tuple[Any, ...]
    ) -> tuple[TerminatorValue, PythonValues]:
        return ReturnedValues(args), ()

    @impl(smt.ConstantBoolOp)
    def run_constant(
        self, interpreter: Interpreter, op: smt.ConstantBoolOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return (bool(op.value),)

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
        return (args.count(False) == 0,)

    @impl(smt.OrOp)
    def run_or(
        self, interpreter: Interpreter, op: smt.OrOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return (args.count(True) > 0,)

    @impl(smt.ImpliesOp)
    def run_implies(
        self, interpreter: Interpreter, op: smt.ImpliesOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(args[0], bool)
        assert isinstance(args[1], bool)
        return (not args[0] or args[1],)

    @impl(smt.XOrOp)
    def run_xor(
        self, interpreter: Interpreter, op: smt.XOrOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return (args.count(True) % 2 == 1,)

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
