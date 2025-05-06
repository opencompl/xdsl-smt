from dataclasses import dataclass, field
from typing import Callable

from xdsl.context import Context
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
from xdsl.ir import Operation, SSAValue, Attribute, Block
from xdsl.dialects.builtin import (
    FunctionType,
)
from xdsl.rewriter import Rewriter


def call_function(func: DefineFunOp, args: list[SSAValue]) -> CallOp:
    """
    Given a function in smt dialect and its args, return CallOp(func, args) with type checking
    """
    func_args = func.body.block.args
    if len(func_args) != len(args):
        raise ValueError(f"Arguments of the call to function {func.fun_name} mismatch")
    for f_arg, arg in zip(func_args, args):
        if f_arg.type != arg.type:
            print(func_args)
            print(args)
            raise ValueError(
                f"Argument of the call to function {func.fun_name} has different type"
            )
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
    if len(callOp.results) != 1:
        raise ValueError(f"Incorrect returned value {func.fun_name}")
    firstOp = FirstOp(callOp.results[0])
    assertOps = assert_result(firstOp.res, bv)
    return [callOp, firstOp] + assertOps


def call_function_and_eq_result_with_effect(
    func: DefineFunOp, args: list[SSAValue], bv: ConstantOp, effect: SSAValue
) -> tuple[list[Operation], EqOp]:
    callOp, callFirstOp = call_function_with_effect(func, args, effect)
    firstOp = FirstOp(callFirstOp.res)
    eqOp = EqOp(firstOp.res, bv.res)
    return [callOp, callFirstOp, firstOp, eqOp], eqOp


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


def insert_argument_instances_to_block_with_effect(
    func: DefineFunOp, int_attr: dict[int, int], block: Block
) -> list[SSAValue]:
    result: list[SSAValue] = []
    assert isinstance(func.body.block.args[-1].type, BoolType)
    curLen = len(block.args)
    for i, arg in enumerate(func.body.block.args[:-1]):
        argType = arg.type
        if i in int_attr:
            constOp = ConstantOp(int_attr[i], get_width_from_type(argType))
            block.add_op(constOp)
            result.append(constOp.res)
        else:
            block.insert_arg(argType, curLen)
            curLen += 1
            result.append(block.args[-1])
    return result


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
    if not isinstance(func.body.block.args[-1].type, BoolType):
        raise ValueError(f"Function {func.fun_name} is not ended with effect")
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


def insert_result_instances_to_block_with_effect(
    func: DefineFunOp, block: Block
) -> list[SSAValue]:
    return_type = func.func_type.outputs.data[0]
    assert isa(return_type, AnyPairType)
    real_return_type = return_type.first
    effect_type = return_type.second
    assert isinstance(effect_type, BoolType)
    block.insert_arg(real_return_type, 0)
    result: list[SSAValue] = [block.args[0]]
    return result


def insert_result_instances_to_block(func: DefineFunOp, block: Block) -> list[SSAValue]:
    return_type = func.func_type.outputs.data[0]
    block.insert_arg(return_type, 0)
    result: list[SSAValue] = [block.args[0]]
    return result


def get_result_instance_with_effect(
    func: DefineFunOp,
) -> tuple[list[Operation], SSAValue]:
    """
    Given a function, construct a list of its returned value by DeclareConstOp
    We assume only the first returned value is useful
    """
    return_type = func.func_type.outputs.data[0]
    assert isa(return_type, PairType)
    declConstOp = DeclareConstOp(return_type.first)
    assert isinstance(return_type.second, BoolType)
    return [declConstOp], declConstOp.res


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
        raise ValueError("cannot compress lst with size 0 to an AndOp")
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
    return func.func_type == func1.func_type


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
        Rewriter.replace_value_with_new_type(func.ret, new_smt_func_type)
    return func


@dataclass
class FunctionCollection:
    """
    This class maintains a map from width(int) -> smt function function
    When the desired function with given width doesn't exist, it generates one
    and returns it as the result
    This class is used when we need several instances of one same function but with different
    possible bit widths.
    """

    main_func: FuncOp
    create_smt: Callable[[FuncOp, int, Context], DefineFunOp]
    ctx: Context
    smt_funcs: dict[int, DefineFunOp] = field(default_factory=dict[int, DefineFunOp])

    def getFunctionByWidth(self, width: int) -> DefineFunOp:
        if width not in self.smt_funcs:
            self.smt_funcs[width] = self.create_smt(self.main_func, width, self.ctx)
        return self.smt_funcs[width]


@dataclass
class TransferFunction:
    """
    This class maintains information about a transfer function before lowering to smt
    """

    transfer_function: FuncOp

    is_abstract_arg: list[bool] = field(init=False)
    """"
    is_abstract_arg[ith] == True -> ith argument of the transfer function is an abstract value
    is_abstract_arg[ith] == False -> ith argument of the transfer function is not an abstract value,
    which maybe a constant value or extra parameters
    """

    name: str = field(init=False)

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

    def __post_init__(self):
        self.name = self.transfer_function.sym_name.data
        is_abstract_arg: list[bool] = []
        func_type = self.transfer_function.function_type
        for func_type_arg, arg in zip(func_type.inputs, self.transfer_function.args):
            assert func_type_arg == arg.type
            is_abstract_arg.append(isinstance(arg.type, AbstractValueType))
        self.is_abstract_arg = is_abstract_arg


@dataclass
class SMTTransferFunction:
    """
    This class maintains information about a transfer function after lowering to SMT
    """

    transfer_function_before_smt: TransferFunction
    is_abstract_arg: list[bool] = field(init=False)
    is_forward: bool = field(init=False)
    operationNo: int = field(init=False)

    transfer_function_name: str
    concrete_function_name: str

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

    precision_util: DefineFunOp | None
    """
    A util function used only in backward precision check for now
    """

    int_attr_arg: list[int] | None

    int_attr_constraint: DefineFunOp | None
    """
    This function maintains the constraint  of integer attributes
    For example, the truncated length should be larger than 0 and less than the total bit width
    """
    transfer_function: DefineFunOp | None = None
    concrete_function: DefineFunOp | None = None

    def __post_init__(self):
        self.is_abstract_arg = self.transfer_function_before_smt.is_abstract_arg
        self.is_forward = self.transfer_function_before_smt.is_forward
        self.operationNo = self.transfer_function_before_smt.operationNo

    def verify(self):
        assert compare_defining_op(self.transfer_function, self.abstract_constraint)
        assert compare_defining_op(self.concrete_function, self.op_constraint)
        if self.transfer_function is not None:
            assert len(self.is_abstract_arg) == len(
                self.transfer_function.body.block.args
            )
        assert self.is_forward ^ (self.operationNo != -1)
