from typing import Callable

from filecheck.filecheck import Check

from .transfer_function_util import replaceAbstractValueWidth
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
from ..utils.transfer_function_util import (
    getArgumentInstances,
    getResultInstance,
    callFunctionAndAssertResult, callFunction, getArgumentWidths, getResultWidth, compress_and_op,
    SMTTransferFunction,FunctionCollection
)

def valid_abstract_domain_check(transfer_function:SMTTransferFunction,
                                domain_constraint:FunctionCollection):
    abstract_func = transfer_function.transfer_function
    abs_op_constraint = transfer_function.abstract_constraint
    abs_arg_ops = getArgumentInstances(abstract_func)
    abs_args = [arg.res for arg in abs_arg_ops]
    is_abstract_arg = transfer_function.is_abstract_arg

    constant_bv_0 = ConstantOp(0, 1)
    constant_bv_1 = ConstantOp(1, 1)

    arg_widths = getArgumentWidths(abstract_func)
    result_width = getResultWidth(abstract_func)

    abs_domain_constraints_ops = []
    for i, abs_arg in enumerate(abs_args):
        if is_abstract_arg[i]:
            abs_domain_constraints_ops += (
                callFunctionAndAssertResult(domain_constraint.getFunctionByWidth(arg_widths[i]), [abs_arg],
                                            constant_bv_1))

    abs_arg_constraints_ops = []
    if abs_op_constraint is not None:
        abs_arg_constraints_ops = callFunctionAndAssertResult(abs_op_constraint, abs_args, constant_bv_1)

    call_abs_func_op = callFunction(abstract_func, abs_args)
    abs_result_domain_invalid_ops=callFunctionAndAssertResult(
        domain_constraint.getFunctionByWidth(result_width),[call_abs_func_op.res],constant_bv_0)
    return (abs_arg_ops+[constant_bv_0, constant_bv_1] + abs_domain_constraints_ops
            +abs_arg_constraints_ops+[call_abs_func_op]+abs_result_domain_invalid_ops)

def int_attr_check(transfer_function:SMTTransferFunction,
                            domain_constraint:FunctionCollection,
                            instance_constraint:FunctionCollection,
                   int_attr:dict[int,int])->list[Operation]:
    if transfer_function.int_attr_constraint is not None:
        int_attr_constraint=transfer_function.int_attr_constraint
        int_attr_constraint_arg_ops = getArgumentInstances(int_attr_constraint, int_attr)
        int_attr_constraint_arg=[arg.res for arg in int_attr_constraint_arg_ops]

        constant_bv_1 = ConstantOp(1, 1)

        call_constraint_ops=callFunctionAndAssertResult(int_attr_constraint,int_attr_constraint_arg, constant_bv_1)
        return int_attr_constraint_arg_ops + [constant_bv_1] +call_constraint_ops +[CheckSatOp()]
    else:
        true_op = ConstantBoolOp(True)
        assert_op=AssertOp(true_op.res)
        return [true_op, assert_op] +[CheckSatOp()]

def forward_soundness_check(transfer_function:SMTTransferFunction,
                            domain_constraint:FunctionCollection,
                            instance_constraint:FunctionCollection,
                            int_attr:dict[int,int]):
    assert transfer_function.is_forward
    abstract_func=transfer_function.transfer_function
    concrete_func=transfer_function.concrete_function
    abs_op_constraint=transfer_function.abstract_constraint
    op_constraint=transfer_function.op_constraint
    is_abstract_arg=transfer_function.is_abstract_arg

    abs_arg_ops=getArgumentInstances(abstract_func, int_attr)
    abs_args=[arg.res for arg in abs_arg_ops]
    crt_arg_ops=getArgumentInstances(concrete_func, int_attr)
    crt_args_with_poison=[arg.res for arg in crt_arg_ops]
    crt_arg_first_ops=[FirstOp(arg) for arg in crt_args_with_poison]
    crt_args=[arg.res for arg in crt_arg_first_ops]

    assert len(abs_args) == len(crt_args)
    arg_widths=getArgumentWidths(concrete_func)
    result_width=getResultWidth(concrete_func)

    constant_bv_0 = ConstantOp(0, 1)
    constant_bv_1 = ConstantOp(1, 1)

    abs_arg_include_crt_arg_constraints_ops=[]
    abs_domain_constraints_ops=[]
    for i, (abs_arg, crt_arg) in enumerate(zip(abs_args, crt_args)):
        if is_abstract_arg[i]:
            abs_arg_include_crt_arg_constraints_ops+=(
                callFunctionAndAssertResult(instance_constraint.getFunctionByWidth(arg_widths[i]),[abs_arg, crt_arg], constant_bv_1))
            abs_domain_constraints_ops+=(
                callFunctionAndAssertResult(domain_constraint.getFunctionByWidth(arg_widths[i]), [abs_arg], constant_bv_1))


    abs_arg_constraints_ops = []
    if abs_op_constraint is not None:
        abs_arg_constraints_ops = callFunctionAndAssertResult(abs_op_constraint, abs_args, constant_bv_1)
    crt_args_constraints_ops = []
    if op_constraint is not None:
        crt_args_constraints_ops = callFunctionAndAssertResult(op_constraint, crt_args, constant_bv_1)

    call_abs_func_op=callFunction(abstract_func, abs_args)
    call_crt_func_op=callFunction(concrete_func, crt_args_with_poison)
    call_crt_first_op = FirstOp(call_crt_func_op.res)

    abs_result_not_include_crt_result_ops=callFunctionAndAssertResult(instance_constraint.getFunctionByWidth(result_width),
                                                                      [call_abs_func_op.res, call_crt_first_op.res], constant_bv_0)

    return (abs_arg_ops + crt_arg_ops +crt_arg_first_ops+ [constant_bv_0, constant_bv_1]+ abs_domain_constraints_ops + abs_arg_include_crt_arg_constraints_ops
            +abs_arg_constraints_ops+crt_args_constraints_ops+[call_abs_func_op, call_crt_func_op, call_crt_first_op]+abs_result_not_include_crt_result_ops + [CheckSatOp()])


def backward_soundness_check(transfer_function: SMTTransferFunction,
                             domain_constraint:FunctionCollection,
                            instance_constraint: FunctionCollection,
                             int_attr:dict[int,int]):
    assert not transfer_function.is_forward
    operationNo=transfer_function.operationNo
    abstract_func = transfer_function.transfer_function
    concrete_func = transfer_function.concrete_function
    abs_op_constraint = transfer_function.abstract_constraint
    op_constraint = transfer_function.op_constraint
    is_abstract_arg = transfer_function.is_abstract_arg

    arg_widths = getArgumentWidths(concrete_func)
    result_width = getResultWidth(concrete_func)

    #replace the only abstract arg in transfer_function with bv with result_width
    assert (sum(is_abstract_arg) == 1)
    abs_arg_idx=is_abstract_arg.index(True)
    old_abs_arg=abstract_func.body.block.args[abs_arg_idx]
    new_abs_arg_type=replaceAbstractValueWidth(old_abs_arg.type, result_width)
    new_abs_arg = abstract_func.body.block.insert_arg(new_abs_arg_type,abs_arg_idx)
    abstract_func.body.block.args[abs_arg_idx+1].replace_by(new_abs_arg)
    abstract_func.body.block.erase_arg(old_abs_arg)

    abs_arg_ops = getArgumentInstances(abstract_func, int_attr)
    abs_args = [arg.res for arg in abs_arg_ops]

    crt_arg_ops = getArgumentInstances(concrete_func, int_attr)
    crt_args_with_poison = [arg.res for arg in crt_arg_ops]
    crt_arg_first_ops = [FirstOp(arg) for arg in crt_args_with_poison]
    crt_args = [arg.res for arg in crt_arg_first_ops]

    constant_bv_0 = ConstantOp(0, 1)
    constant_bv_1 = ConstantOp(1, 1)

    call_abs_func_op = callFunction(abstract_func, abs_args)
    call_crt_func_op = callFunction(concrete_func, crt_args_with_poison)
    call_crt_func_first_op = FirstOp(call_crt_func_op.res)

    abs_domain_constraints_ops = callFunctionAndAssertResult(domain_constraint.getFunctionByWidth(result_width),[abs_args[0]], constant_bv_1)

    abs_arg_include_crt_res_constraint_ops=callFunctionAndAssertResult(instance_constraint.getFunctionByWidth(result_width),
                                                                       [abs_args[0], call_crt_func_first_op.res], constant_bv_1)

    abs_arg_constraints_ops = []
    if abs_op_constraint is not None:
        abs_arg_constraints_ops = callFunctionAndAssertResult(abs_op_constraint, abs_args, constant_bv_1)
    crt_args_constraints_ops = []
    if op_constraint is not None:
        crt_args_constraints_ops = callFunctionAndAssertResult(op_constraint, crt_args, constant_bv_1)

    abs_result_not_include_crt_arg_constraint_ops=callFunctionAndAssertResult(instance_constraint.getFunctionByWidth(arg_widths[operationNo]),
                                                                              [call_abs_func_op.res, crt_args[operationNo]], constant_bv_0)

    return (abs_arg_ops + crt_arg_ops + [constant_bv_0, constant_bv_1] + [call_abs_func_op, call_crt_func_op, call_crt_func_first_op] +
            abs_domain_constraints_ops +abs_arg_include_crt_res_constraint_ops + abs_arg_constraints_ops + crt_args_constraints_ops + abs_result_not_include_crt_arg_constraint_ops+[CheckSatOp()])


def forward_precision_check(transfer_function: SMTTransferFunction,
                            domain_constraint:FunctionCollection,
                            instance_constraint: FunctionCollection):
    assert transfer_function.is_forward

def backward_precision_check(transfer_function: SMTTransferFunction,
                             domain_constraint:FunctionCollection,
                            instance_constraint: FunctionCollection):
    assert not transfer_function.is_forward

def counterexample_check(counter_func:FuncOp, smt_counter_func:DefineFunOp,
                         domain_constraint:FunctionCollection, int_attr:dict[int,int]):
    is_abstract_arg:list[bool] = [isinstance(arg, AbstractValueType) for arg in counter_func.args]
    arg_ops=getArgumentInstances(smt_counter_func, int_attr)
    args=[arg.res for arg in arg_ops]
    arg_widths=getArgumentWidths(smt_counter_func)

    constant_bv_1 = ConstantOp(1, 1)

    abs_domain_constraints_ops = []
    for i, arg in enumerate(args):
        if is_abstract_arg[i]:
            abs_domain_constraints_ops += (
                callFunctionAndAssertResult(domain_constraint.getFunctionByWidth(arg_widths[i]), [arg],
                                            constant_bv_1))
    call_counterexample_func_ops=callFunctionAndAssertResult(smt_counter_func,args,constant_bv_1)

    return arg_ops + [constant_bv_1] + abs_domain_constraints_ops+call_counterexample_func_ops+ [CheckSatOp()]