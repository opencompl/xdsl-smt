from typing import Callable

from ..dialects.smt_dialect import (
    SMTDialect,
    DefineFunOp,
    DeclareConstOp,
    CallOp,
    AssertOp,
    CheckSatOp,
    EqOp,
    ConstantBoolOp,
    ImpliesOp,
    ForallOp,
    AndOp,
    YieldOp,
)
from ..dialects.smt_bitvector_dialect import (
    SMTBitVectorDialect,
    ConstantOp,
    BitVectorType,
)
from ..dialects.smt_utils_dialect import FirstOp, PairOp, PairType
from ..dialects.index_dialect import Index
from ..dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.ir.core import BlockArgument
from xdsl.dialects.builtin import (
    Builtin,
    ModuleOp,
    IntegerAttr,
    IntegerType,
    i1,
    FunctionType,
    Region,
    Block,
)
from xdsl.dialects.func import Func, FuncOp, Return, Call
from ..dialects.transfer import Transfer, AbstractValueType
from xdsl.dialects.arith import Arith
import xdsl.dialects.comb as comb
from xdsl.ir import Operation, SSAValue, Attribute
from ..passes.lower_to_smt.lower_to_smt import LowerToSMT, integer_poison_type_lowerer
from ..passes.lower_to_smt import (
    func_to_smt_patterns,
)
from xdsl_smt.semantics import transfer_semantics
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.transfer_semantics import (
    transfer_semantics,
    abstract_value_type_lowerer,
    transfer_integer_type_lowerer,
)
from xdsl_smt.semantics.comb_semantics import comb_semantics
import sys as sys

#Given a function and its args, return CallOp(func, args) with type checking
def callFunction(func:DefineFunOp, args:list[SSAValue]) -> CallOp:
    func_args=func.body.block.args
    assert len(func_args) == len(args)
    for f_arg, arg in zip(func_args, args):
        if f_arg.type != arg.type:
            print(func.fun_name)
            print(func_args)
            print(args)
        assert f_arg.type == arg.type
    callOp = CallOp.get(func.results[0], args)
    return callOp

def assertResult(result:SSAValue, bv:ConstantOp) -> list[Operation]:
    eqOp = EqOp.get(result, bv.res)
    assertOp=AssertOp.get(eqOp.res)
    return [eqOp, assertOp]

#Given a function, its argument and a constant bv, assert the return value by CallOp(func, args)
#equals to the bv
def callFunctionAndAssertResult(func: DefineFunOp, args:list[SSAValue], bv:ConstantOp) -> list[Operation]:
    callOp=callFunction(func, args)
    firstOp = FirstOp(callOp.res)
    assertOps=assertResult(firstOp.res, bv)
    return [callOp, firstOp] +assertOps

#Given a function, return a list of argument instances
def getArgumentInstances(func:DefineFunOp, int_attr:dict[int,int]) ->list[DeclareConstOp|ConstantOp]:
    result=[]
    for i,arg in enumerate(func.body.block.args):
        argType=arg.type
        if i in int_attr:
            result.append(ConstantOp(int_attr[i],getWdithFromType(argType)))
        else:
            result.append(DeclareConstOp(argType))
    return result

#Given a function, return an instance of its return value
def getResultInstance(func:DefineFunOp) -> list[DeclareConstOp]:
    return_type = func.func_type.outputs.data[0]
    return [DeclareConstOp(return_type)]

def getWdithFromType(ty:Attribute) -> int:
    if isinstance(ty, PairType):
        ty=ty.first
    if isinstance(ty, BitVectorType):
        return ty.width.data
    assert False

def replaceAbstractValueWidth(abs_val_ty:PairType, new_width:int) -> PairType:
    types=[]
    while isinstance(abs_val_ty, PairType):
        types.append(abs_val_ty.first)
        abs_val_ty=abs_val_ty.second
    types.append(abs_val_ty)
    for i in range(len(types)):
        if isinstance(types[i], BitVectorType):
            types[i]=BitVectorType.from_int(new_width)
    resultType=types.pop()
    while len(types) > 0:
        resultType=PairType(types.pop(), resultType)
    return resultType

def getArgumentWidths(func:DefineFunOp) -> list[int]:
    return [getWdithFromType(arg.type) for arg in func.body.block.args]

def getResultWidth(func:DefineFunOp) -> int:
    return getWdithFromType(func.func_type.outputs.data[0])

def compress_and_op(lst) -> (SSAValue, list[Operation]):
    if len(lst) == 0:
        assert False and "cannot compress lst with size 0 to an AndOp"
    elif len(lst) == 1:
        return (lst[0].res, [])
    else:
        new_ops: list[AndOp] = [AndOp(lst[0].results[0], lst[1].results[0])]
        for i in range(2, len(lst)):
            new_ops.append(AndOp(new_ops[-1].res, lst[i].results[0]))
        return (new_ops[-1].res, new_ops)

def compareDefiningOp(func:DefineFunOp, func1:DefineFunOp) -> bool:
    for arg, arg1 in zip(func.body.block.args, func1.body.block.args):
        if arg.type != arg1.type:
            return False
    return func.func_type.outputs.data[0] == func1.func_type.outputs.data[0]

# This class maintains a map from width(int) -> a function
# When the desired function with given width doesn't exist, it generates one
# and returns it as the result
class FunctionCollection:
    main_func:FuncOp=None
    smt_funcs:dict[int, DefineFunOp]={}
    create_smt:Callable[[FuncOp, int], DefineFunOp]=None

    def __init__(self, func:FuncOp, create_smt:Callable[[FuncOp, int], DefineFunOp]):
        self.main_func=func
        self.create_smt=create_smt
        self.smt_funcs={}

    def getFunctionByWidth(self, width:int) -> DefineFunOp:
        if width not in self.smt_funcs:
            self.smt_funcs[width]=self.create_smt(self.main_func,width)
        return self.smt_funcs[width]


#This class maintains information about a transfer function
class TransferFunction:
    is_abstract_arg:list[bool]=[]
    name:str = ""
    is_forward:bool=True
    replace_int_attr=False
    operationNo:int=-1
    transfer_function:FuncOp=None

    def __init__(self, transfer_function:FuncOp, is_forward:bool=True, operationNo:int=-1, replace_int_attr:bool=False):
        self.name=transfer_function.sym_name.data
        self.is_forward=is_forward
        self.operationNo=operationNo
        is_abstract_arg=[]
        self.transfer_function=transfer_function
        func_type = transfer_function.function_type
        for func_type_arg, arg in zip(func_type.inputs, transfer_function.args):
            is_abstract_arg.append(isinstance(arg.type, AbstractValueType))
        self.is_abstract_arg = is_abstract_arg
        self.replace_int_attr=replace_int_attr

#This class maintains information about a transfer function after lowered to SMT
class SMTTransferFunction:
    is_abstract_arg:list[bool]=[]
    is_forward:bool=True
    operationNo: int = -1
    transfer_function_name:str=None
    transfer_function:DefineFunOp=None
    concrete_function_name:str=None
    concrete_function:DefineFunOp=None
    abstract_constraint:DefineFunOp=None
    op_constraint:DefineFunOp=None
    soundness_counterexample: DefineFunOp = None
    int_attr_arg:list[int] =None
    int_attr_constraint:DefineFunOp =None

    def __init__(self, transfer_function_name:str, transfer_function:DefineFunOp, tfRecord:dict[str, TransferFunction],
                 concrete_function_name:str,
                 concrete_function:DefineFunOp,
                 abstract_constraint:DefineFunOp, op_constraint:DefineFunOp, soundness_counterexample:DefineFunOp,
                 int_attr_arg:list[int], int_attr_constraint:DefineFunOp):
        self.transfer_function_name=transfer_function_name
        self.concrete_function_name=concrete_function_name
        assert self.transfer_function_name in tfRecord
        tf = tfRecord[self.transfer_function_name]
        self.transfer_function=transfer_function
        self.is_forward = tf.is_forward
        self.is_abstract_arg = tf.is_abstract_arg
        self.concrete_function=concrete_function
        self.abstract_constraint=abstract_constraint
        self.op_constraint=op_constraint
        self.operationNo=tf.operationNo
        self.soundness_counterexample=soundness_counterexample
        self.int_attr_arg=int_attr_arg
        self.int_attr_constraint=int_attr_constraint

    def verify(self):
        assert compareDefiningOp(self.transfer_function, self.abstract_constraint)
        assert compareDefiningOp(self.concrete_function, self.op_constraint)
        assert len(self.is_abstract_arg) == len(self.transfer_function.body.block.args)
        assert self.is_forward ^ (self.operationNo !=-1)