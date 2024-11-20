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
from xdsl.ir import Operation, SSAValue, Attribute, Block
from xdsl.dialects.builtin import (
    FunctionType,
)


# Given a function and its args, return CallOp(func, args) with type checking
def callFunction(func: DefineFunOp, args: list[SSAValue]) -> CallOp:
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


# Given a function and its args, return CallOp(func, args) with type checking
def callFunctionWithEffect(
    func: DefineFunOp, args: list[SSAValue], effect: SSAValue
) -> tuple[CallOp, FirstOp]:
    func_args = func.body.block.args
    new_args = args + [effect]
    assert len(func_args) == len(new_args)
    for f_arg, arg in zip(func_args, new_args):
        if f_arg.type != arg.type:
            print(func.fun_name)
            print(func_args)
            print(new_args)
        assert f_arg.type == arg.type
    callOp = CallOp.get(func.results[0], new_args)
    assert len(callOp.res) == 1
    callOpFirst = FirstOp(callOp.res[0])
    callOpSecond = SecondOp(callOp.res[0])
    assert isinstance(callOpSecond.res.type, BoolType)
    return callOp, callOpFirst


def assertResult(result: SSAValue, bv: ConstantOp) -> list[Operation]:
    eqOp = EqOp.get(result, bv.res)
    assertOp = AssertOp.get(eqOp.res)
    return [eqOp, assertOp]


# Given a function, its argument and a constant bv, assert the return value by CallOp(func, args)
# equals to the bv
def callFunctionAndAssertResult(
    func: DefineFunOp, args: list[SSAValue], bv: ConstantOp
) -> list[Operation]:
    callOp = callFunction(func, args)
    assert len(callOp.results) == 1
    firstOp = FirstOp(callOp.results[0])
    assertOps = assertResult(firstOp.res, bv)
    return [callOp, firstOp] + assertOps


def callFunctionAndEqResultWithEffect(
    func: DefineFunOp, args: list[SSAValue], bv: ConstantOp, effect: SSAValue
) -> tuple[list[Operation], EqOp]:
    callOp, callFirstOp = callFunctionWithEffect(func, args, effect)
    firstOp = FirstOp(callFirstOp.res)
    eqOp = EqOp(firstOp.res, bv.res)
    return [callOp, callFirstOp, firstOp, eqOp], eqOp


def callFunctionAndAssertResultWithEffect(
    func: DefineFunOp, args: list[SSAValue], bv: ConstantOp, effect: SSAValue
) -> list[Operation]:
    callOp, callFirstOp = callFunctionWithEffect(func, args, effect)
    firstOp = FirstOp(callFirstOp.res)
    assertOps = assertResult(firstOp.res, bv)
    return [callOp, callFirstOp, firstOp] + assertOps


def insertArgumentInstancesToBlockWithEffect(
    func: DefineFunOp, int_attr: dict[int, int], block: Block
) -> list[SSAValue]:
    result: list[SSAValue] = []
    assert isinstance(func.body.block.args[-1].type, BoolType)
    curLen = len(block.args)
    for i, arg in enumerate(func.body.block.args[:-1]):
        argType = arg.type
        if i in int_attr:
            constOp = ConstantOp(int_attr[i], getWidthFromType(argType))
            block.add_op(constOp)
            result.append(constOp.res)
        else:
            block.insert_arg(argType, curLen)
            curLen += 1
            result.append(block.args[-1])
    return result


# Given a function, return a list of argument instances
def getArgumentInstancesWithEffect(
    func: DefineFunOp, int_attr: dict[int, int]
) -> list[DeclareConstOp | ConstantOp]:
    result: list[DeclareConstOp | ConstantOp] = []
    # ignore last effect arg
    assert isinstance(func.body.block.args[-1].type, BoolType)
    for i, arg in enumerate(func.body.block.args[:-1]):
        argType = arg.type
        if i in int_attr:
            result.append(ConstantOp(int_attr[i], getWidthFromType(argType)))
        else:
            result.append(DeclareConstOp(argType))
    return result


# Given a function, return a list of argument instances
def getArgumentInstances(
    func: DefineFunOp, int_attr: dict[int, int]
) -> list[DeclareConstOp | ConstantOp]:
    result: list[DeclareConstOp | ConstantOp] = []
    for i, arg in enumerate(func.body.block.args):
        argType = arg.type
        if i in int_attr:
            result.append(ConstantOp(int_attr[i], getWidthFromType(argType)))
        else:
            result.append(DeclareConstOp(argType))
    return result


def insertResultInstancesToBlock(func: DefineFunOp, block: Block) -> list[SSAValue]:
    return_type = func.func_type.outputs.data[0]
    block.insert_arg(return_type, 0)
    result: list[SSAValue] = [block.args[0]]
    return result


# Given a function, return an instance of its return value
def getResultInstance(func: DefineFunOp) -> list[DeclareConstOp]:
    return_type = func.func_type.outputs.data[0]
    return [DeclareConstOp(return_type)]


def getWidthFromType(ty: Attribute) -> int:
    while isa(ty, AnyPairType):
        assert isinstance(ty.first, Attribute)
        ty = ty.first
    if isinstance(ty, BitVectorType):
        return ty.width.data
    assert False


def replaceAbstractValueWidth(
    abs_val_ty: AnyPairType | Attribute, new_width: int
) -> AnyPairType:
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


def getArgumentWidthsWithEffect(func: DefineFunOp) -> list[int]:
    # ignore last effect
    return [getWidthFromType(arg.type) for arg in func.body.block.args[:-1]]


def getArgumentWidths(func: DefineFunOp) -> list[int]:
    return [getWidthFromType(arg.type) for arg in func.body.block.args]


def getResultWidth(func: DefineFunOp) -> int:
    return getWidthFromType(func.func_type.outputs.data[0])


def compress_and_op(lst: list[Operation]) -> tuple[SSAValue, list[Operation]]:
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


def compareDefiningOp(func: DefineFunOp | None, func1: DefineFunOp | None) -> bool:
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


def fixDefiningOpReturnType(func: DefineFunOp) -> DefineFunOp:
    smt_func_type = func.func_type
    ret_val_type = [ret.type for ret in func.return_values]
    if smt_func_type != ret_val_type:
        new_smt_func_type = FunctionType.from_lists(
            smt_func_type.inputs.data, ret_val_type
        )
        func.ret.type = new_smt_func_type
    return func


# This class maintains a map from width(int) -> a function
# When the desired function with given width doesn't exist, it generates one
# and returns it as the result
class FunctionCollection:
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


# This class maintains information about a transfer function
class TransferFunction:
    is_abstract_arg: list[bool] = []
    name: str = ""
    is_forward: bool = True
    replace_int_attr: bool = False
    operationNo: int = -1
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


# This class maintains information about a transfer function after lowered to SMT
class SMTTransferFunction:
    is_abstract_arg: list[bool] = []
    is_forward: bool = True
    operationNo: int = -1
    transfer_function_name: str
    transfer_function: DefineFunOp | None = None
    concrete_function_name: str
    concrete_function: DefineFunOp | None = None
    abstract_constraint: DefineFunOp | None
    op_constraint: DefineFunOp | None
    soundness_counterexample: DefineFunOp | None
    int_attr_arg: list[int] | None
    int_attr_constraint: DefineFunOp | None

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
        assert compareDefiningOp(self.transfer_function, self.abstract_constraint)
        assert compareDefiningOp(self.concrete_function, self.op_constraint)
        if self.transfer_function is not None:
            assert len(self.is_abstract_arg) == len(
                self.transfer_function.body.block.args
            )
        assert self.is_forward ^ (self.operationNo != -1)
