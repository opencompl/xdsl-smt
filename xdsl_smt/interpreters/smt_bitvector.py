from typing import Any, Callable, cast

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


def compute(
    f: Callable[[int], int] | Callable[[int, int], int],
    args: tuple[Any],
    *,
    width: int | None = None
) -> tuple[bv.BitVectorAttr]:
    assert all(isinstance(arg, bv.BitVectorAttr) for arg in args)
    result_width = width if width is not None else args[0].type.width.data
    return (
        bv.BitVectorAttr(
            f(*(arg.value.data for arg in args)) & ((1 << result_width) - 1),
            bv.BitVectorType(result_width),
        ),
    )


@register_impls
class SMTBitVectorFunctions(InterpreterFunctions):
    @impl_attr(BitVectorType)
    def bv_attr_value(
        self, interpreter: Interpreter, attr: Attribute, attr_type: BitVectorType
    ) -> bv.BitVectorAttr:
        interpreter.interpreter_assert(isinstance(attr, IntegerAttr))
        attr = cast(IntegerAttr, attr)
        return bv.BitVectorAttr(
            attr.value.data % attr_type.width.data, attr_type.width.data
        )

    @impl(bv.ConstantOp)
    def run_bv_constant(
        self, interpreter: Interpreter, op: bv.ConstantOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return (op.value,)

    @impl(bv.AddOp)
    def run_bv_add(
        self, interpreter: Interpreter, op: bv.AddOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda l, r: l + r, args)

    @impl(bv.AndOp)
    def run_bv_and(
        self, interpreter: Interpreter, op: bv.AndOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda l, r: l & r, args)

    @impl(bv.AShrOp)
    def run_bv_ashr(
        self, interpreter: Interpreter, op: bv.AShrOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda l, r: l >> r, args)

    @impl(bv.LShrOp)
    def run_bv_lshr(
        self, interpreter: Interpreter, op: bv.LShrOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda l, r: l << r, args)

    @impl(bv.MulOp)
    def run_bv_mul(
        self, interpreter: Interpreter, op: bv.MulOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda l, r: l * r, args)

    @impl(bv.NegOp)
    def run_bv_neg(
        self, interpreter: Interpreter, op: bv.NegOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda x: -x, args)

    @impl(bv.NotOp)
    def run_bv_not(
        self, interpreter: Interpreter, op: bv.NotOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda x: ~x, args)

    @impl(bv.OrOp)
    def run_bv_or(
        self, interpreter: Interpreter, op: bv.OrOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda l, r: l | r, args)
