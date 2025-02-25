from xdsl.dialects.builtin import i1, IntegerAttr

from ..dialects.transfer import (
    NegOp,
    CmpOp,
    AndOp,
    OrOp,
    XorOp,
    AddOp,
    SubOp,
    CountLOneOp,
    CountLZeroOp,
    CountROneOp,
    CountRZeroOp,
    # SetHighBitsOp,
    # SetLowBitsOp,
    # GetBitWidthOp,
    # UMulOverflowOp,
    # SMinOp,
    # SMaxOp,
    # UMinOp,
    # UMaxOp,
    ShlOp,
    LShrOp,
    SelectOp,
    UnaryOp,
    Constant,
    GetAllOnesOp,
    MulOp,
    SetLowBitsOp,
    SetHighBitsOp,
    TransIntegerType,
)
from typing import TypeVar, Generic, Callable
from xdsl.ir import Operation, SSAValue
import xdsl.dialects.arith as arith
from xdsl_smt.utils.random import Random

T = TypeVar("T")


class Collection(Generic[T]):
    """
    This class implements a data structure called Collection.
    It supports O(1) insert, delete and retrieve a random element.
    """

    lst: list[T]
    lst_len: int
    ele_to_index: dict[T, int]
    random: Random

    def __init__(self, lst: list[T], random: Random):
        self.lst = lst
        self.ele_to_index = {}
        self.lst_len = len(lst)
        for i, ele in enumerate(lst):
            self.ele_to_index[ele] = i
        self.random = random

    def add(self, ele: T):
        self.lst.append(ele)
        self.lst_len += 1
        self.ele_to_index[ele] = self.lst_len

    def remove(self, ele: T):
        if ele in self.ele_to_index:
            idx = self.ele_to_index[ele]
            self.lst[idx] = self.lst[-1]
            self.lst.pop()
            self.lst_len -= 1
            del self.ele_to_index[ele]

    def size(self):
        return self.lst_len

    def get_random_element(self) -> T | None:
        if self.lst_len != 0:
            return self.random.choice(self.lst)
        return None

    def get_all_elements(self) -> tuple[T, ...]:
        return tuple(self.lst)

    def get_random_element_if(self, predicate: Callable[[T], bool]) -> T | None:
        idx = self.random.randint(0, self.lst_len - 1)
        for _ in range(self.lst_len):
            if predicate(self.lst[idx]):
                return self.lst[idx]
            idx += 1
            idx %= self.lst_len
        return None


basic_int_ops: list[type[Operation]] = [
    NegOp,
    AndOp,
    OrOp,
    XorOp,
    AddOp,
    # SubOp,
    # SelectOp,
]


full_int_ops: list[type[Operation]] = [
    NegOp,
    AndOp,
    OrOp,
    XorOp,
    AddOp,
    SubOp,
    SelectOp,
    LShrOp,
    ShlOp,
    CountLOneOp,
    CountLZeroOp,
    CountROneOp,
    CountRZeroOp,
]

basic_i1_ops: list[type[Operation]] = [arith.AndIOp, arith.OrIOp, arith.XOrIOp, CmpOp]


def is_constant_constructor(constants: list[int]) -> Callable[[SSAValue], bool]:
    is_constant: Callable[[SSAValue], bool] = lambda val=SSAValue: (
        isinstance(val, Constant) and val.value.value.data in constants
    )
    return is_constant


is_zero_or_one: Callable[[SSAValue], bool] = is_constant_constructor([0, 1])

is_zero: Callable[[SSAValue], bool] = is_constant_constructor([0])

is_one: Callable[[SSAValue], bool] = is_constant_constructor([1])

is_true: Callable[[SSAValue], bool] = lambda val=SSAValue: (
    isinstance(val, arith.ConstantOp)
    and isinstance(val.value, IntegerAttr)
    and val.value.value.data == 1
)

is_false: Callable[[SSAValue], bool] = lambda val=SSAValue: (
    isinstance(val, arith.ConstantOp)
    and isinstance(val.value, IntegerAttr)
    and val.value.value.data == 0
)

is_constant_bool: Callable[[SSAValue], bool] = lambda val=SSAValue: isinstance(
    val, arith.ConstantOp
)


def is_allones(val: SSAValue) -> bool:
    return isinstance(val.owner, GetAllOnesOp)


def is_get_bitwidth(val: SSAValue) -> bool:
    return isinstance(val.owner, GetAllOnesOp)


def is_zero_or_allones(val: SSAValue) -> bool:
    return is_allones(val) or is_zero(val)


def is_one_or_allones(val: SSAValue) -> bool:
    return is_allones(val) or is_one(val)


def is_zero_or_one_or_allones(val: SSAValue) -> bool:
    return is_allones(val) or is_zero_or_one(val)


def no_constraint(val: SSAValue) -> bool:
    return False


"""
Two dictionaries maintains optimizations on operand selection.
True value means we should not use that SSAValue as the operand
"""

optimize_operands_selection: dict[type[Operation], Callable[[SSAValue], bool]] = {
    # Transfer operations
    NegOp: is_zero_or_allones,
    AddOp: is_zero,
    SubOp: is_zero,
    MulOp: is_zero_or_one,
    AndOp: is_zero_or_allones,
    OrOp: is_zero_or_allones,
    XorOp: is_zero_or_allones,
    CountLZeroOp: is_zero_or_one_or_allones,
    CountRZeroOp: is_zero_or_one_or_allones,
    CountLOneOp: is_zero_or_one_or_allones,
    CountROneOp: is_zero_or_one_or_allones,
    ShlOp: is_zero_or_allones,
    LShrOp: is_zero_or_allones,
    # arith operations
    arith.AndIOp: is_constant_bool,
    arith.OrIOp: is_constant_bool,
    arith.XOrIOp: is_false,
}


"""
Complex selection mechanism.
For each operand we should have a predicate.
"""

optimize_complex_operands_selection: dict[
    type[Operation], list[Callable[[SSAValue], bool]]
] = {
    SelectOp: [is_constant_bool, no_constraint, no_constraint],
    SetLowBitsOp: [is_one_or_allones, is_zero_or_allones],
    SetHighBitsOp: [is_allones, is_zero_or_allones],
}

"""
Idempotent property means we should not use the same operand for both operand.
"""

idempotent_operations: set[type[Operation]] = {
    # Transfer operations
    SubOp,
    AndOp,
    OrOp,
    XorOp,
    CmpOp,
    # Special case for true and false branch
    SelectOp,
    # arith operations
    arith.AndIOp,
    arith.OrIOp,
    arith.XOrIOp,
}


class SynthesizerContext:
    random: Random
    cmp_flags: list[int]
    i1_ops: Collection[type[Operation]]
    int_ops: Collection[type[Operation]]
    commutative: bool = False
    idempotent: bool = False
    skip_trivial: bool = False

    def __init__(self, random: Random):
        self.random = random
        self.cmp_flags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.i1_ops = Collection(basic_i1_ops, self.random)
        self.int_ops = Collection(basic_int_ops, self.random)

    def use_basic_int_ops(self):
        self.int_ops = Collection(basic_int_ops, self.random)

    def use_full_int_ops(self):
        self.int_ops = Collection(full_int_ops, self.random)

    def get_available_i1_ops(self) -> tuple[type[Operation], ...]:
        return self.i1_ops.get_all_elements()

    def get_available_int_ops(self) -> tuple[type[Operation], ...]:
        return self.int_ops.get_all_elements()

    def set_cmp_flags(self, cmp_flags: list[int]):
        assert len(cmp_flags) != 0
        for flag in cmp_flags:
            assert 0 <= flag and flag <= 9
        self.cmp_flags = cmp_flags

    def get_random_class(self) -> Random:
        return self.random

    def get_constraint(self, op: type[Operation]) -> Callable[[SSAValue], bool]:
        if self.skip_trivial:
            return optimize_operands_selection.get(op, no_constraint)
        return no_constraint

    def is_idempotent(self, op: type[Operation]) -> bool:
        if self.idempotent:
            return op in idempotent_operations
        return False

    def select_operand(
        self, vals: list[SSAValue], constraint: Callable[[SSAValue], bool]
    ) -> SSAValue:
        current_pos = self.random.randint(0, len(vals) - 1)
        for _ in range(len(vals)):
            if not constraint(vals[current_pos]):
                return vals[current_pos]
            current_pos += 1
            current_pos %= len(vals)
        raise ValueError("Cannot find any matched operand")

    def select_two_operand(
        self,
        vals: list[SSAValue],
        constraint1: Callable[[SSAValue], bool],
        constraint2: Callable[[SSAValue], bool] | None = None,
        is_idempotent: bool = False,
    ) -> tuple[SSAValue, SSAValue]:
        val1 = self.select_operand(vals, constraint1)
        if constraint2 is None:
            constraint2 = constraint1
        if is_idempotent:
            constraint2 = lambda val=SSAValue: constraint1(val) and val != val1
        val2 = self.select_operand(vals, constraint2)
        return val1, val2

    def build_i1_op(
        self,
        result_type: type[Operation],
        int_vals: list[SSAValue],
        i1_vals: list[SSAValue],
    ) -> Operation:
        if result_type == CmpOp:
            val1, val2 = self.select_two_operand(
                int_vals,
                self.get_constraint(result_type),
                is_idempotent=self.is_idempotent(result_type),
            )
            return CmpOp(
                val1,
                val2,
                self.random.choice(self.cmp_flags),
            )
        assert result_type is not None
        val1, val2 = self.select_two_operand(
            i1_vals,
            self.get_constraint(result_type),
            is_idempotent=self.is_idempotent(result_type),
        )
        result = result_type(
            val1,  # pyright: ignore [reportCallIssue]
            val2,
        )
        assert isinstance(result, Operation)
        return result

    def build_int_op(
        self,
        result_type: type[Operation],
        int_vals: list[SSAValue],
        i1_vals: list[SSAValue],
    ) -> Operation:
        assert result_type is not None
        if result_type == SelectOp:
            (
                cond_constraint,
                true_constraint,
                false_constraint,
            ) = optimize_complex_operands_selection[SelectOp]
            cond = self.select_operand(i1_vals, cond_constraint)
            true_val, false_val = self.select_two_operand(
                int_vals,
                true_constraint,
                false_constraint,
                is_idempotent=self.is_idempotent(result_type),
            )
            return SelectOp(cond, true_val, false_val)
        elif issubclass(result_type, UnaryOp):
            val = self.select_operand(int_vals, self.get_constraint(result_type))
            return result_type(val)
        elif self.skip_trivial and result_type in optimize_complex_operands_selection:
            constraint1, constraint2 = optimize_complex_operands_selection[result_type]
            val1, val2 = self.select_two_operand(
                int_vals,
                constraint1,
                constraint2,
                is_idempotent=self.is_idempotent(result_type),
            )
        else:
            val1, val2 = self.select_two_operand(
                int_vals,
                self.get_constraint(result_type),
                is_idempotent=self.is_idempotent(result_type),
            )
        result = result_type(
            val1,  # pyright: ignore [reportCallIssue]
            val2,
        )
        assert isinstance(result, Operation)
        return result

    def get_random_i1_op(
        self,
        int_vals: list[SSAValue],
        i1_vals: list[SSAValue],
    ) -> Operation:
        result_type = self.i1_ops.get_random_element()
        assert result_type is not None
        return self.build_i1_op(result_type, int_vals, i1_vals)

    def get_random_int_op(
        self,
        int_vals: list[SSAValue],
        i1_vals: list[SSAValue],
    ) -> Operation:
        result_type = self.int_ops.get_random_element()
        assert result_type is not None
        return self.build_int_op(result_type, int_vals, i1_vals)

    def get_random_i1_op_except(
        self,
        int_vals: list[SSAValue],
        i1_vals: list[SSAValue],
        except_op: Operation,
    ) -> Operation:
        result_type = self.i1_ops.get_random_element_if(
            lambda op_ty=type[Operation]: op_ty != type(except_op)
        )
        assert result_type is not None
        return self.build_i1_op(result_type, int_vals, i1_vals)

    def get_random_int_op_except(
        self,
        int_vals: list[SSAValue],
        i1_vals: list[SSAValue],
        except_op: Operation,
    ) -> Operation:
        result_type = self.int_ops.get_random_element_if(
            lambda op_ty=type[Operation]: op_ty != type(except_op)
        )
        assert result_type is not None
        return self.build_int_op(result_type, int_vals, i1_vals)

    def replace_operand(
        self, op: Operation, int_vals: list[SSAValue], i1_vals: list[SSAValue]
    ):
        ith = self.random.randint(0, len(op.operands) - 1)
        if not self.skip_trivial:
            # NOTICE: consider not the same value?
            if isinstance(op.operands[ith].type, TransIntegerType):
                op.operands[ith] = self.select_operand(int_vals, no_constraint)
            elif op.operands[ith].type == i1:
                op.operands[ith] = self.select_operand(i1_vals, no_constraint)
            else:
                raise ValueError(
                    "unknown type when replacing operand: " + str(op.operands[ith].type)
                )
            return True
        op_type = type(op)
        constraint: Callable[[SSAValue], bool] = self.get_constraint(op_type)
        if op_type in optimize_complex_operands_selection:
            constraint = optimize_complex_operands_selection[op_type][ith]

        is_idempotent = self.is_idempotent(op_type)
        new_constraint: Callable[[SSAValue], bool] | None = None
        if is_idempotent:
            # not the condition variable of select op
            if op_type == SelectOp and ith != 0:
                # 3 - ith -> given ith, select the other branch
                new_constraint = lambda val=SSAValue: (
                    constraint(val) and val != op.operands[3 - ith]
                )
            else:
                new_constraint = lambda val=SSAValue: (
                    constraint(val) and val != op.operands[1 - ith]
                )
        if isinstance(op.operands[ith].type, TransIntegerType):
            op.operands[ith] = self.select_operand(
                int_vals, constraint if new_constraint is None else new_constraint
            )
        elif op.operands[ith].type == i1:
            op.operands[ith] = self.select_operand(
                i1_vals, constraint if new_constraint is None else new_constraint
            )
        else:
            raise ValueError(
                "unknown type when replacing operand: " + str(op.operands[ith].type)
            )
        return True
