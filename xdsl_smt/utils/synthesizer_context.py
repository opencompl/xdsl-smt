from xdsl.dialects.builtin import i1, IntegerAttr
from xdsl.dialects.func import FuncOp

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
    GetBitWidthOp,
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

    def get_weighted_random_element(self, weights: dict[T, int]) -> T | None:
        if self.lst_len != 0:
            return self.random.choice_weighted(self.lst, weights=weights)
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
    MulOp,
    CountLOneOp,
    CountLZeroOp,
    CountROneOp,
    CountRZeroOp,
    # SetHighBitsOp,
    # SetLowBitsOp,
]

full_i1_ops: list[type[Operation]] = [arith.AndIOp, arith.OrIOp, arith.XOrIOp, CmpOp]

basic_i1_ops: list[type[Operation]] = [CmpOp]


def is_constant_constructor(constants: list[int]) -> Callable[[SSAValue], bool]:
    is_constant: Callable[[SSAValue], bool] = lambda val=SSAValue: (
        isinstance(val.owner, Constant) and val.owner.value.value.data in constants
    )
    return is_constant


is_zero_or_one: Callable[[SSAValue], bool] = is_constant_constructor([0, 1])

is_zero: Callable[[SSAValue], bool] = is_constant_constructor([0])

is_one: Callable[[SSAValue], bool] = is_constant_constructor([1])

is_true: Callable[[SSAValue], bool] = lambda val=SSAValue: (
    isinstance(val.owner, arith.ConstantOp)
    and isinstance(val.owner.value, IntegerAttr)
    and val.owner.value.value.data == 1
)

is_false: Callable[[SSAValue], bool] = lambda val=SSAValue: (
    isinstance(val.owner, arith.ConstantOp)
    and isinstance(val.owner.value, IntegerAttr)
    and val.owner.value.value.data == 0
)

is_constant_bool: Callable[[SSAValue], bool] = lambda val=SSAValue: isinstance(
    val.owner, arith.ConstantOp
)


def is_allones(val: SSAValue) -> bool:
    return isinstance(val.owner, GetAllOnesOp)


def is_get_bitwidth(val: SSAValue) -> bool:
    return isinstance(val.owner, GetBitWidthOp)


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
    arith.XOrIOp: is_constant_bool,
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
    i1_weights: dict[type[Operation], int]
    int_weights: dict[type[Operation], int]
    weighted: bool
    commutative: bool = False
    idempotent: bool = True
    skip_trivial: bool = True

    def __init__(
        self,
        random: Random,
    ):
        self.random = random
        self.cmp_flags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.i1_ops = Collection(basic_i1_ops, self.random)
        self.int_ops = Collection(basic_int_ops, self.random)
        self.i1_weights = {key: 1 for key in basic_i1_ops}
        self.int_weights = {key: 1 for key in basic_int_ops}
        self.weighted = True

    def use_basic_int_ops(self):
        self.int_ops = Collection(basic_int_ops, self.random)
        self.int_weights = {key: 1 for key in basic_int_ops}

    def use_full_int_ops(self):
        self.int_ops = Collection(full_int_ops, self.random)
        self.int_weights = {key: 1 for key in full_int_ops}

    def use_basic_i1_ops(self):
        self.i1_ops = Collection(basic_i1_ops, self.random)
        self.i1_weights = {key: 1 for key in basic_i1_ops}

    def use_full_i1_ops(self):
        self.i1_ops = Collection(full_i1_ops, self.random)
        self.i1_weights = {key: 1 for key in full_i1_ops}

    def get_available_i1_ops(self) -> tuple[type[Operation], ...]:
        return self.i1_ops.get_all_elements()

    def get_available_int_ops(self) -> tuple[type[Operation], ...]:
        return self.int_ops.get_all_elements()

    def update_i1_weights(self, frequency: dict[type[Operation], int]):
        self.i1_weights = {key: 1 for key in self.i1_ops.get_all_elements()}
        for key in frequency:
            assert key in self.i1_weights
            self.i1_weights[key] = frequency[key] + 1

    def update_int_weights(self, frequency: dict[type[Operation], int]):
        self.int_weights = {key: 1 for key in self.int_ops.get_all_elements()}
        for key in frequency:
            assert key in self.int_weights
            self.int_weights[key] = frequency[key] + 1

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
        self,
        vals: list[SSAValue],
        constraint: Callable[[SSAValue], bool],
        exclude_val: SSAValue | None = None,
    ) -> SSAValue | None:
        current_pos = self.random.randint(0, len(vals) - 1)
        for _ in range(len(vals)):
            if not constraint(vals[current_pos]) and vals[current_pos] != exclude_val:
                return vals[current_pos]
            current_pos += 1
            current_pos %= len(vals)
        return None

    def select_two_operand(
        self,
        vals: list[SSAValue],
        constraint1: Callable[[SSAValue], bool],
        constraint2: Callable[[SSAValue], bool] | None = None,
        is_idempotent: bool = False,
    ) -> tuple[SSAValue | None, SSAValue | None]:
        val1 = self.select_operand(vals, constraint1)
        if val1 is None:
            return None, None
        if constraint2 is None:
            constraint2 = constraint1
        if is_idempotent:
            val2 = self.select_operand(vals, constraint2, val1)
        else:
            val2 = self.select_operand(vals, constraint2)
        return val1, val2

    def build_i1_op(
        self,
        result_type: type[Operation],
        int_vals: list[SSAValue],
        i1_vals: list[SSAValue],
    ) -> Operation | None:
        if result_type == CmpOp:
            val1, val2 = self.select_two_operand(
                int_vals,
                self.get_constraint(result_type),
                is_idempotent=self.is_idempotent(result_type),
            )
            if val1 is None or val2 is None:
                return None
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
        if val1 is None or val2 is None:
            return None
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
    ) -> Operation | None:
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
            if cond is None or true_val is None or false_val is None:
                return None
            return SelectOp(cond, true_val, false_val)
        elif issubclass(result_type, UnaryOp):
            val = self.select_operand(int_vals, self.get_constraint(result_type))
            if val is None:
                return None
            return result_type(val)  # pyright: ignore [reportCallIssue]
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

        if val1 is None or val2 is None:
            return None
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
    ) -> Operation | None:
        if self.weighted:
            result_type = self.i1_ops.get_weighted_random_element(self.i1_weights)
        else:
            result_type = self.i1_ops.get_random_element()
        assert result_type is not None
        return self.build_i1_op(result_type, int_vals, i1_vals)

    def get_random_int_op(
        self,
        int_vals: list[SSAValue],
        i1_vals: list[SSAValue],
    ) -> Operation | None:
        if self.weighted:
            result_type = self.int_ops.get_weighted_random_element(self.int_weights)
        else:
            result_type = self.int_ops.get_random_element()
        assert result_type is not None
        return self.build_int_op(result_type, int_vals, i1_vals)

    def get_random_i1_op_except(
        self,
        int_vals: list[SSAValue],
        i1_vals: list[SSAValue],
        except_op: Operation,
    ) -> Operation | None:
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
    ) -> Operation | None:
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
                val = self.select_operand(int_vals, no_constraint)
                if val is None:
                    return False
                op.operands[ith] = val
            elif op.operands[ith].type == i1:
                val = self.select_operand(int_vals, no_constraint)
                if val is None:
                    return False
                op.operands[ith] = val
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
            val = self.select_operand(
                int_vals, constraint if new_constraint is None else new_constraint
            )
            if val is None:
                return False
            op.operands[ith] = val
        elif op.operands[ith].type == i1:
            val = self.select_operand(
                i1_vals, constraint if new_constraint is None else new_constraint
            )
            if val is None:
                return False
            op.operands[ith] = val
        else:
            raise ValueError(
                "unknown type when replacing operand: " + str(op.operands[ith].type)
            )
        return True

    @staticmethod
    def count_op_frequency(
        func: FuncOp,
    ) -> tuple[dict[type[Operation], int], dict[type[Operation], int]]:
        freq_int: dict[type[Operation], int] = {}
        freq_i1: dict[type[Operation], int] = {}
        for op in func.body.block.ops:
            ty = type(op)
            if ty in full_int_ops:
                freq_int[ty] = freq_int.get(ty, 0) + 1
            if ty in full_i1_ops:
                freq_i1[ty] = freq_i1.get(ty, 0) + 1
        return freq_int, freq_i1
