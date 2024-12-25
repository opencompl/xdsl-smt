from typing import Callable

from xdsl.context import MLContext
from xdsl.utils.hints import isa
from ..dialects.smt_dialect import (
    DefineFunOp,
    DeclareConstOp,
    CallOp,
    AssertOp,
    EqOp,
    AndOp,
    BoolType,
)
from ..dialects.smt_bitvector_dialect import (
    ConstantOp,
    BitVectorType,
)
from ..dialects.smt_utils_dialect import FirstOp, PairType, SecondOp, AnyPairType
from xdsl.dialects.func import FuncOp
from ..dialects.transfer import AbstractValueType
from xdsl.ir import Operation, SSAValue, Attribute
from xdsl.dialects.builtin import (
    FunctionType,
)


def call_function(func: DefineFunOp, args: list[SSAValue]) -> CallOp:
    """
    Given a function in smt dialect and its args, return CallOp(func, args) with type checking
    """
    func_args = func.body.block.args
    assert len(func_args) == len(args)
    for f_arg, arg in zip(func_args, args):
        if f_arg.type != arg.type:
            print(func.fun_name)
            print(func_args)
            print(args)
        assert f_arg.type == arg.type
    callOp = CallOp.get(func.results[0], args)
    return callOp


def call_function_with_effect(
    func: DefineFunOp, args: list[SSAValue], effect: SSAValue
) -> tuple[CallOp, FirstOp]:
    """
    In current design, a FuncOp is lowered to DefineFunOp with receiving and returning the global effect.
    However, transfer functions don't use that field.
    This function is a shortcut for calling a DefineFunOp with adding the effect to arguments,
    and removing the effect from the returned value.
    """
    new_args = args + [effect]
    callOp = call_function(func, new_args)
    assert len(callOp.res) == 1
    callOpFirst = FirstOp(callOp.res[0])
    callOpSecond = SecondOp(callOp.res[0])
    # Assume the global effect has a bool type
    assert isinstance(callOpSecond.res.type, BoolType)
    return callOp, callOpFirst


# Given a SSAValue and bv constant, asserts the SSAValue equals to the given constant
def assert_result(result: SSAValue, bv: ConstantOp) -> list[Operation]:
    eqOp = EqOp.get(result, bv.res)
    assertOp = AssertOp.get(eqOp.res)
    return [eqOp, assertOp]


def call_function_and_assert_result(
    func: DefineFunOp, args: list[SSAValue], bv: ConstantOp
) -> list[Operation]:
    """
    Given a function, its argument and a constant bv, assert the return value by CallOp(func, args)
    equals to the bv
    """
    callOp = call_function(func, args)
    assert len(callOp.results) == 1
    firstOp = FirstOp(callOp.results[0])
    assertOps = assert_result(firstOp.res, bv)
    return [callOp, firstOp] + assertOps


def call_function_and_assert_result_with_effect(
    func: DefineFunOp, args: list[SSAValue], bv: ConstantOp, effect: SSAValue
) -> list[Operation]:
    """
    Given a function with global effect, its argument and a constant bv, assert the return value of function calling
    equals to the bv
    """
    callOp, callFirstOp = call_function_with_effect(func, args, effect)
    firstOp = FirstOp(callFirstOp.res)
    assertOps = assert_result(firstOp.res, bv)
    return [callOp, callFirstOp, firstOp] + assertOps


def get_argument_instances_with_effect(
    func: DefineFunOp, int_attr: dict[int, int]
) -> list[DeclareConstOp | ConstantOp]:
    """
    Given a function, construct a list of argument instances
    by DeclareConstOp or ConstantOp except for the last effect argument
    Some operations require certain arguments must be constant (e.g. the length of truncated integer),
    and this information is maintained in int_attr
    """
    result: list[DeclareConstOp | ConstantOp] = []
    # ignore last effect arg
    assert isinstance(func.body.block.args[-1].type, BoolType)
    for i, arg in enumerate(func.body.block.args[:-1]):
        argType = arg.type
        if i in int_attr:
            result.append(ConstantOp(int_attr[i], get_width_from_type(argType)))
        else:
            result.append(DeclareConstOp(argType))
    return result


def get_argument_instances(
    func: DefineFunOp, int_attr: dict[int, int]
) -> list[DeclareConstOp | ConstantOp]:
    """
    Given a function, construct a list of argument instances by DeclareConstOp or ConstantOp
    Some operations require certain arguments must be constant (e.g. the length of truncated integer),
    and this information is maintained in int_attr
    """
    result: list[DeclareConstOp | ConstantOp] = []
    for i, arg in enumerate(func.body.block.args):
        argType = arg.type
        if i in int_attr:
            result.append(ConstantOp(int_attr[i], get_width_from_type(argType)))
        else:
            result.append(DeclareConstOp(argType))
    return result


def get_result_instance(func: DefineFunOp) -> list[DeclareConstOp]:
    """
    Given a function, construct a list of its returned value by DeclareConstOp
    We assume only the first returned value is useful
    """
    return_type = func.func_type.outputs.data[0]
    return [DeclareConstOp(return_type)]


def get_width_from_type(ty: Attribute) -> int:
    """
    Given a bit vector type or a pair type including a bit vector,
    returns the bit width of that bit vector
    """
    while isa(ty, AnyPairType):
        assert isinstance(ty.first, Attribute)
        ty = ty.first
    if isinstance(ty, BitVectorType):
        return ty.width.data
    assert False


def replace_abstract_value_width(
    abs_val_ty: AnyPairType | Attribute, new_width: int
) -> AnyPairType:
    """
    Given a pair type and a bit width, this function replaces all bit vector type
    in the input pair type with a new bit vector type with the new bit width
    """
    types: list[Attribute] = []
    while isa(abs_val_ty, AnyPairType):
        assert isinstance(abs_val_ty.first, Attribute)
        types.append(abs_val_ty.first)
        assert isinstance(abs_val_ty.second, Attribute)
        abs_val_ty = abs_val_ty.second
    types.append(abs_val_ty)
    for i in range(len(types)):
        if isinstance(types[i], BitVectorType):
            types[i] = BitVectorType.from_int(new_width)
    resultType = types.pop()
    while len(types) > 0:
        resultType = PairType(types.pop(), resultType)
    assert isa(resultType, AnyPairType)
    return resultType


def get_argument_widths_with_effect(func: DefineFunOp) -> list[int]:
    """
    Given a smt function, returns a list of bit width for every argument of the function except for the last effect
    """
    # ignore last effect
    return [get_width_from_type(arg.type) for arg in func.body.block.args[:-1]]


def get_argument_widths(func: DefineFunOp) -> list[int]:
    """
    Given a smt function, returns a list of bit width for every argument of the function
    """
    return [get_width_from_type(arg.type) for arg in func.body.block.args]


def get_result_width(func: DefineFunOp) -> int:
    """
    Given a smt function, returns the bit width of the returned value
    """
    return get_width_from_type(func.func_type.outputs.data[0])


def compress_and_op(lst: list[Operation]) -> tuple[SSAValue, list[Operation]]:
    """
    Given a list of operations returning bool type, this function performs
    bool and operation on all operations, and returns a tuple of the final combined
    result and a list of constructed and operations
    """
    if len(lst) == 0:
        assert False and "cannot compress lst with size 0 to an AndOp"
    elif len(lst) == 1:
        empty_result: list[Operation] = []
        return (lst[0].results[0], empty_result)
    else:
        new_ops: list[Operation] = [AndOp(lst[0].results[0], lst[1].results[0])]
        for i in range(2, len(lst)):
            new_ops.append(AndOp(new_ops[-1].results[0], lst[i].results[0]))
        return (new_ops[-1].results[0], new_ops)


def compare_defining_op(func: DefineFunOp | None, func1: DefineFunOp | None) -> bool:
    """
    Given two smt functions, returns true if both are None or have the same function type
    """
    func_none: bool = func is None
    func1_none: bool = func1 is None
    if func_none ^ func1_none:
        return False
    if func_none or func1_none:
        return True
    for arg, arg1 in zip(func.body.block.args, func1.body.block.args):
        if arg.type != arg1.type:
            return False
    return func.func_type.outputs.data[0] == func1.func_type.outputs.data[0]


def fix_defining_op_return_type(func: DefineFunOp) -> DefineFunOp:
    """
    Given a smt function, if the type of returned value doesn't match
    the type of function signature, it replaces the output type of function signature
    with the actual returned type
    """
    smt_func_type = func.func_type
    ret_val_type = [ret.type for ret in func.return_values]
    if smt_func_type != ret_val_type:
        new_smt_func_type = FunctionType.from_lists(
            smt_func_type.inputs.data, ret_val_type
        )
        func.ret.type = new_smt_func_type
    return func


class FunctionCollection:
    """
    This class maintains a map from width(int) -> smt function function
    When the desired function with given width doesn't exist, it generates one
    and returns it as the result
    This class is used when we need several instances of one same function but with different
    possible bit widths.
    """

    main_func: FuncOp
    smt_funcs: dict[int, DefineFunOp] = {}
    create_smt: Callable[[FuncOp, int, MLContext], DefineFunOp]
    ctx: MLContext

    def __init__(
        self,
        func: FuncOp,
        create_smt: Callable[[FuncOp, int, MLContext], DefineFunOp],
        ctx: MLContext,
    ):
        self.main_func = func
        self.create_smt = create_smt
        self.smt_funcs = {}
        self.ctx = ctx

    def getFunctionByWidth(self, width: int) -> DefineFunOp:
        if width not in self.smt_funcs:
            self.smt_funcs[width] = self.create_smt(self.main_func, width, self.ctx)
        return self.smt_funcs[width]


class TransferFunction:
    """
    This class maintains information about a transfer function before lowering to smt
    """

    is_abstract_arg: list[bool] = []
    """"
    is_abstract_arg[ith] == True -> ith argument of the transfer function is an abstract value
    is_abstract_arg[ith] == False -> ith argument of the transfer function is not an abstract value,
    which maybe a constant value or extra parameters
    """

    name: str = ""

    is_forward: bool = True
    """
    indicates if this transfer function applies forwards or backwards
    """

    replace_int_attr: bool = False
    """
    This field indicates if some arguments should be replaced by a constant such as
    the length of truncated integer
    """

    operationNo: int = -1
    """
    When the transfer function applies backwards, this field indicates which argument it applies to
    """

    transfer_function: FuncOp

    def __init__(
        self,
        transfer_function: FuncOp,
        is_forward: bool = True,
        operationNo: int = -1,
        replace_int_attr: bool = False,
    ):
        self.name = transfer_function.sym_name.data
        self.is_forward = is_forward
        self.operationNo = operationNo
        is_abstract_arg: list[bool] = []
        self.transfer_function = transfer_function
        func_type = transfer_function.function_type
        for func_type_arg, arg in zip(func_type.inputs, transfer_function.args):
            assert func_type_arg == arg.type
            is_abstract_arg.append(isinstance(arg.type, AbstractValueType))
        self.is_abstract_arg = is_abstract_arg
        self.replace_int_attr = replace_int_attr


class SMTTransferFunction:
    """
    This class maintains information about a transfer function after lowering to SMT
    """

    is_abstract_arg: list[bool] = []
    is_forward: bool = True
    operationNo: int = -1
    transfer_function_name: str
    transfer_function: DefineFunOp | None = None
    concrete_function_name: str
    concrete_function: DefineFunOp | None = None

    abstract_constraint: DefineFunOp | None
    """
    This function describes constraints applied on arguments of the transfer function.
    For example, transfer functions in demanded bits use known bits information as extra parameters,
    we have to make sure all known bits are in valid domain.
    """

    op_constraint: DefineFunOp | None
    """
    This function describes constraints applied on arguments of the concrete function.
    For example, SHL requires the shifting amount must be in a valid range
    """

    soundness_counterexample: DefineFunOp | None
    """
    Except for the basic soundness property checker, if there are other scenarios making the transfer function unsound
    """

    int_attr_arg: list[int] | None

    int_attr_constraint: DefineFunOp | None
    """
    This function maintains the constraint  of integer attributes
    For example, the truncated length should be larger than 0 and less than the total bit width
    """

    def __init__(
        self,
        transfer_function_name: str,
        transfer_function: DefineFunOp | None,
        tfRecord: dict[str, TransferFunction],
        concrete_function_name: str,
        concrete_function: DefineFunOp | None,
        abstract_constraint: DefineFunOp | None,
        op_constraint: DefineFunOp | None,
        soundness_counterexample: DefineFunOp | None,
        int_attr_arg: list[int] | None,
        int_attr_constraint: DefineFunOp | None,
    ):
        self.transfer_function_name = transfer_function_name
        self.concrete_function_name = concrete_function_name
        assert self.transfer_function_name in tfRecord
        tf = tfRecord[self.transfer_function_name]
        self.transfer_function = transfer_function
        self.is_forward = tf.is_forward
        self.is_abstract_arg = tf.is_abstract_arg
        self.concrete_function = concrete_function
        self.abstract_constraint = abstract_constraint
        self.op_constraint = op_constraint
        self.operationNo = tf.operationNo
        self.soundness_counterexample = soundness_counterexample
        self.int_attr_arg = int_attr_arg
        self.int_attr_constraint = int_attr_constraint

    def verify(self):
        assert compare_defining_op(self.transfer_function, self.abstract_constraint)
        assert compare_defining_op(self.concrete_function, self.op_constraint)
        if self.transfer_function is not None:
            assert len(self.is_abstract_arg) == len(
                self.transfer_function.body.block.args
            )
        assert self.is_forward ^ (self.operationNo != -1)
