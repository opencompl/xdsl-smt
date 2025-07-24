from typing import Any

from xdsl.ir import Attribute

from xdsl.utils.hints import isa
from xdsl.dialects.builtin import ArrayAttr
from xdsl_smt.dialects import smt_utils_dialect as smt_utils
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    impl_attr,
    impl,
    register_impls,
)


@register_impls
class SMTUtilsFunctions(InterpreterFunctions):
    @impl_attr(smt_utils.PairType)
    def bool_attr_value(
        self,
        interpreter: Interpreter,
        attr: Attribute,
        attr_type: smt_utils.PairType[Attribute, Attribute],
    ) -> tuple[Any, Any]:
        assert isa(attr, ArrayAttr[Attribute])
        val_first = interpreter.value_for_attribute(attr.data[0], attr_type.first)
        val_second = interpreter.value_for_attribute(attr.data[1], attr_type.second)
        return (val_first, val_second)

    @impl(smt_utils.FirstOp)
    def run_first(
        self, interpreter: Interpreter, op: smt_utils.FirstOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        return (args[0][0],)

    @impl(smt_utils.SecondOp)
    def run_second(
        self, interpreter: Interpreter, op: smt_utils.SecondOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        return (args[0][1],)

    @impl(smt_utils.PairOp)
    def run_pair(
        self, interpreter: Interpreter, op: smt_utils.PairOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert len(args) == 2
        return ((args[0], args[1]),)
