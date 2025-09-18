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
from xdsl.utils.hints import isa


def compute(
    f: Callable[[int], int] | Callable[[int, int], int],
    args: tuple[Any],
    *,
    width: int | None = None,
) -> tuple[bv.BitVectorAttr]:
    assert all(isinstance(arg, bv.BitVectorAttr) for arg in args)
    result_width = width if width is not None else args[0].type.width.data
    return (
        bv.BitVectorAttr(
            f(*(arg.value.data for arg in args)) & ((1 << result_width) - 1),
            bv.BitVectorType(result_width),
        ),
    )


def to_signed(value: int, width: int) -> int:
    """Convert an unsigned integer to a signed integer."""
    if value >= (1 << (width - 1)):
        return value - (1 << width)
    return value


def to_unsigned(value: int, width: int) -> int:
    """Convert a signed integer to an unsigned integer."""
    if value < 0:
        return value + (1 << width)
    return value


@register_impls
class SMTBitVectorFunctions(InterpreterFunctions):
    @impl_attr(BitVectorType)
    def bv_attr_value(
        self, interpreter: Interpreter, attr: Attribute, attr_type: BitVectorType
    ) -> bv.BitVectorAttr:
        interpreter.interpreter_assert(isinstance(attr, IntegerAttr))
        attr = cast(IntegerAttr, attr)
        return bv.BitVectorAttr(
            attr.value.data % (1 << attr_type.width.data), attr_type.width.data
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
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        width = args[0].type.width.data
        lhs = to_signed(args[0].value.data, width)
        rhs = to_signed(args[1].value.data, width)

        # Arithmetic right shift
        if rhs < 0:
            return (bv.BitVectorAttr(0, width),)
        if rhs >= width:
            if lhs >= 0:
                return (bv.BitVectorAttr(0, width),)
            else:
                return (bv.BitVectorAttr((1 << width) - 1, width),)
        result = lhs >> rhs
        return (bv.BitVectorAttr(to_unsigned(result, width), width),)

    @impl(bv.LShrOp)
    def run_bv_lshr(
        self, interpreter: Interpreter, op: bv.LShrOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda l, r: l >> r, args)

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

    @impl(bv.XorOp)
    def run_bv_xor(
        self, interpreter: Interpreter, op: bv.XorOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda l, r: l ^ r, args)

    @impl(bv.OrOp)
    def run_bv_or(
        self, interpreter: Interpreter, op: bv.OrOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda l, r: l | r, args)

    @impl(bv.SModOp)
    def run_bv_smod(
        self, interpreter: Interpreter, op: bv.SModOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        width = args[0].type.width.data
        lhs = to_signed(args[0].value.data, width)
        rhs = to_signed(args[1].value.data, width)

        if rhs == 0:
            result = to_unsigned(lhs, width)
        else:
            result = to_unsigned(to_unsigned(lhs % rhs, width), width)
        return (bv.BitVectorAttr(result, width),)

    @impl(bv.URemOp)
    def run_bv_urem(
        self, interpreter: Interpreter, op: bv.URemOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        lhs = args[0].value.data
        rhs = args[1].value.data

        if rhs == 0:
            result = lhs
        else:
            result = lhs % rhs
        return (bv.BitVectorAttr(result, args[0].type.width.data),)

    @impl(bv.SRemOp)
    def run_bv_srem(
        self, interpreter: Interpreter, op: bv.SRemOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        width = args[0].type.width.data
        lhs = to_signed(args[0].value.data, width)
        rhs = to_signed(args[1].value.data, width)

        if rhs == 0:
            result = lhs
        else:
            if lhs < 0 and rhs < 0:
                result = to_unsigned(lhs % rhs, width)
            elif lhs < 0:
                result = lhs % rhs
                if result > 0:
                    result -= rhs
            elif rhs < 0:
                result = lhs % -rhs
            else:
                result = lhs % rhs
        return (bv.BitVectorAttr(to_unsigned(result, width), width),)

    @impl(bv.UDivOp)
    def run_bv_udiv(
        self, interpreter: Interpreter, op: bv.UDivOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        width = args[0].type.width.data
        lhs = args[0].value.data
        rhs = args[1].value.data

        if rhs == 0:
            return (bv.BitVectorAttr(to_unsigned(-1, width), width),)

        return (bv.BitVectorAttr(lhs // rhs, args[0].type.width.data),)

    @impl(bv.SDivOp)
    def run_bv_sdiv(
        self, interpreter: Interpreter, op: bv.SDivOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        width = args[0].type.width.data
        lhs = to_signed(args[0].value.data, width)
        rhs = to_signed(args[1].value.data, width)

        if rhs == 0:
            return (bv.BitVectorAttr(to_unsigned(-1, width), width),)

        return (
            bv.BitVectorAttr(to_unsigned(lhs // rhs, width), args[0].type.width.data),
        )

    @impl(bv.CmpOp)
    def run_bv_cmp(
        self, interpreter: Interpreter, op: bv.CmpOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        lhs = args[0].value.data
        rhs = args[1].value.data
        signed_lhs = to_signed(lhs, args[0].type.width.data)
        signed_rhs = to_signed(rhs, args[1].type.width.data)

        match op.pred.value.data:
            case 0:
                return (signed_lhs < signed_rhs,)
            case 1:
                return (signed_lhs <= signed_rhs,)
            case 2:
                return (signed_lhs > signed_rhs,)
            case 3:
                return (signed_lhs >= signed_rhs,)
            case 4:
                return (lhs < rhs,)
            case 5:
                return (lhs <= rhs,)
            case 6:
                return (lhs > rhs,)
            case 7:
                return (lhs >= rhs,)
            case _:
                raise ValueError(f"Unknown predicate {op.pred.value.data}")

    @impl(bv.SgeOp)
    def run_bv_sge(
        self, interpreter: Interpreter, op: bv.SgeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        lhs = args[0].value.data
        rhs = args[1].value.data
        signed_lhs = to_signed(lhs, args[0].type.width.data)
        signed_rhs = to_signed(rhs, args[1].type.width.data)
        return (signed_lhs >= signed_rhs,)

    @impl(bv.SgtOp)
    def run_bv_sgt(
        self, interpreter: Interpreter, op: bv.SgtOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        lhs = args[0].value.data
        rhs = args[1].value.data
        signed_lhs = to_signed(lhs, args[0].type.width.data)
        signed_rhs = to_signed(rhs, args[1].type.width.data)
        return (signed_lhs > signed_rhs,)

    @impl(bv.SltOp)
    def run_bv_slt(
        self, interpreter: Interpreter, op: bv.SltOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        lhs = args[0].value.data
        rhs = args[1].value.data
        signed_lhs = to_signed(lhs, args[0].type.width.data)
        signed_rhs = to_signed(rhs, args[1].type.width.data)
        return (signed_lhs < signed_rhs,)

    @impl(bv.SleOp)
    def run_bv_sle(
        self, interpreter: Interpreter, op: bv.SleOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        lhs = args[0].value.data
        rhs = args[1].value.data
        signed_lhs = to_signed(lhs, args[0].type.width.data)
        signed_rhs = to_signed(rhs, args[1].type.width.data)
        return (signed_lhs <= signed_rhs,)

    @impl(bv.UltOp)
    def run_bv_ult(
        self, interpreter: Interpreter, op: bv.UltOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        lhs = args[0].value.data
        rhs = args[1].value.data
        return (lhs < rhs,)

    @impl(bv.UleOp)
    def run_bv_ule(
        self, interpreter: Interpreter, op: bv.UleOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        lhs = args[0].value.data
        rhs = args[1].value.data
        return (lhs <= rhs,)

    @impl(bv.UgtOp)
    def run_bv_ugt(
        self, interpreter: Interpreter, op: bv.UgtOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        lhs = args[0].value.data
        rhs = args[1].value.data
        return (lhs > rhs,)

    @impl(bv.UgeOp)
    def run_bv_uge(
        self, interpreter: Interpreter, op: bv.UgeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isa(args, tuple[bv.BitVectorAttr, ...])
        lhs = args[0].value.data
        rhs = args[1].value.data
        return (lhs >= rhs,)

    @impl(bv.ShlOp)
    def run_bv_shl(
        self, interpreter: Interpreter, op: bv.ShlOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return compute(lambda l, r: l << r, args)
