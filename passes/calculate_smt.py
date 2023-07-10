from dataclasses import dataclass
from functools import singledispatch

from xdsl.passes import ModulePass

from xdsl.ir import MLContext, OpResult, BlockArgument

from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.dialects import builtin

from utils.integer_to_z3 import *
from utils.trans_interpreter_smt import *

from z3 import ULT, UGE, ULE, UGT, And, Or, Xor

opToSMTFunc = {
    "transfer.add": (lambda solver, a, b: BitVecRef.__add__(a, b)),
    "transfer.sub": (lambda solver, a, b: BitVecRef.__sub__(a, b)),
    "transfer.mul": (lambda solver, a, b: BitVecRef.__mul__(a, b)),
    "transfer.umul_overflow": (lambda solver, a, b: umul_overflow(a, b, solver)),
    "transfer.get_bit_width": (lambda solver, a: BitVecVal(a.size(), a.size())),
    "transfer.countl_zero": (lambda solver, a: count_lzeros(a, solver, a.size())),
    "transfer.countl_one": (lambda solver, a: count_lones(a, solver, a.size())),
    "transfer.countr_zero": (lambda solver, a: count_rzeros(a, solver, a.size())),
    "transfer.countr_one": (lambda solver, a: count_rones(a, solver, a.size())),
    "transfer.smin": (lambda solver, a, b: smin(a, b, solver)),
    "transfer.umin": (lambda solver, a, b: umin(a, b, solver)),
    "transfer.smax": (lambda solver, a, b: smax(a, b, solver)),
    "transfer.umax": (lambda solver, a, b: umax(a, b, solver)),
    "transfer.get_low_bits": (lambda solver, a, b: get_low_bits(a, b)),
    "transfer.set_high_bits": (lambda solver, a, b: set_high_bits(a, b)),
    "transfer.neg": (lambda solver, a: BitVecRef.__invert__(a)),
    "transfer.and": (lambda solver, a, b: BitVecRef.__and__(a, b)),
    "transfer.or": (lambda solver, a, b: BitVecRef.__or__(a, b)),
    "transfer.xor": (lambda solver, a, b: BitVecRef.__xor__(a, b)),
    "transfer.cmp": [
        (lambda solver, a, b: BitVecRef.__eq__(a, b)),
        (lambda solver, a, b: BitVecRef.__ne__(a, b)),
        (lambda solver, a, b: BitVecRef.__lt__(a, b)),
        (lambda solver, a, b: BitVecRef.__le__(a, b)),
        (lambda solver, a, b: BitVecRef.__gt__(a, b)),
        (lambda solver, a, b: BitVecRef.__ge__(a, b)),
        (lambda solver, a, b: ULT(a, b)),
        (lambda solver, a, b: ULE(a, b)),
        (lambda solver, a, b: UGT(a, b)),
        (lambda solver, a, b: UGE(a, b)),
    ],
    "transfer.make": (lambda solver, *operands: operands),
    "transfer.get": (lambda solver, operands, index: operands[index]),
    "transfer.constant": (
        lambda solver, bit_vec, val: get_constant_with_bit_vector(val, bit_vec)
    ),
    "arith.andi": (lambda solver, a, b: And(a, b)),
    "arith.ori": (lambda solver, a, b: Or(a, b)),
    "arith.xori": (lambda solver, a, b: Xor(a, b)),
    "arith.select": (lambda solver, cond, a, b: If(cond, a, b)),
}

funcCall = {}

resultToSMTValue = {}


def get_whole_name(op: Operation):
    parent_op = op.parent_op()
    if parent_op is not None:
        assert isinstance(parent_op, FuncOp)
        return parent_op.sym_name.data + "." + op.results[0].name_hint
    return None


def get_smt_val(op):
    whole_name = None
    if isinstance(op, OpResult):
        op = op.op
        whole_name = get_whole_name(op)
    elif isinstance(op, BlockArgument):
        parent_func = op.block.parent_op()
        whole_name = parent_func.sym_name.data + "." + op.name_hint
    if whole_name not in resultToSMTValue:
        print(whole_name)
        print(op)
    assert whole_name in resultToSMTValue
    return resultToSMTValue[whole_name]


@singledispatch
def to_smt(op, solver, func_name_to_func):
    if op.name not in opToSMTFunc:
        print(op)
        assert False and "not supported operation"
    if len(op.results) > 0:
        whole_name = get_whole_name(op)
        if whole_name not in resultToSMTValue:
            func = opToSMTFunc[op.name]
            operands = [get_smt_val(operand) for operand in op.operands]
            operands = [solver] + operands
            res = func(*operands)
            resultToSMTValue[whole_name] = res
        return resultToSMTValue[whole_name]


@to_smt.register
def _(op: transfer.CmpOp, solver, func_name_to_func):
    whole_name = get_whole_name(op)
    if whole_name not in resultToSMTValue:
        predicate = op.attributes["predicate"].value.data
        func = opToSMTFunc[op.name][predicate]
        operands = [get_smt_val(operand) for operand in op.operands]
        operands = [solver] + operands
        res = func(*operands)
        resultToSMTValue[whole_name] = res
    return resultToSMTValue[whole_name]


@to_smt.register
def _(op: transfer.GetOp, solver, func_name_to_func):
    whole_name = get_whole_name(op)
    if whole_name not in resultToSMTValue:
        operand = get_smt_val(op.operands[0])
        index = op.attributes["index"].value.data
        func = opToSMTFunc[op.name]
        resultToSMTValue[whole_name] = func(solver, operand, index)
    return resultToSMTValue[whole_name]


@to_smt.register
def _(op: transfer.MakeOp, solver, func_name_to_func):
    whole_name = get_whole_name(op)
    if whole_name not in resultToSMTValue:
        operands = [get_smt_val(operand) for operand in op.operands]
        func = opToSMTFunc[op.name]
        result = func(solver, *operands)
        resultToSMTValue[whole_name] = result
    return resultToSMTValue[whole_name]


@to_smt.register
def _(op: transfer.Constant, solver, func_name_to_func):
    whole_name = get_whole_name(op)
    if whole_name not in resultToSMTValue:
        bit_vec = get_smt_val(op.operands[0])
        value = op.attributes["value"].value.data
        func = opToSMTFunc[op.name]
        resultToSMTValue[whole_name] = func(solver, bit_vec, value)
    return resultToSMTValue[whole_name]


@to_smt.register
def _(op: Call, solver, func_name_to_func):
    callee = op.callee.string_value()
    if callee not in funcCall:
        parse_function_to_python(
            func_name_to_func[callee], func_name_to_func, opToSMTFunc, funcCall
        )
    whole_name = get_whole_name(op)
    if whole_name not in resultToSMTValue:
        operands = [get_smt_val(operand) for operand in op.operands]
        func = funcCall[callee]
        resultToSMTValue[whole_name] = func(solver, *operands)
    return resultToSMTValue[whole_name]


# Special handle on return case
# Save the returned value with function name
@to_smt.register
def _(op: Return, solver, func_name_to_func):
    whole_name = op.parent_op().sym_name.data + ".return"
    if whole_name not in resultToSMTValue:
        smt_val = get_smt_val(op.operands[0])
        resultToSMTValue[whole_name] = smt_val
    return resultToSMTValue[whole_name]


def init():
    global resultToSMTValue
    resultToSMTValue = {}


@dataclass
class AssignAttributes(RewritePattern):
    def __init__(self, func_name_to_func, solver, width):
        self.solver = solver
        self.funcNameToFunc = func_name_to_func
        self.width = width

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if isinstance(op, FuncOp):
            for arg in op.args:
                whole_name = op.sym_name.data + "." + arg.name_hint
                if whole_name not in resultToSMTValue:
                    tmp_args = None
                    if isinstance(arg.typ, transfer.AbstractValueType):
                        tmp_args = []
                        for j in range(arg.typ.get_num_fields()):
                            tmp_args.append(
                                BitVec(arg.name_hint + "field" + str(j), self.width)
                            )
                    elif isinstance(arg.typ, builtin.IntegerType):
                        tmp_args = BitVec(arg.name_hint, arg.typ.width.data)
                    elif isinstance(arg.typ, builtin.IndexType):
                        tmp_args = BitVec(arg.name_hint, self.width)
                    elif isinstance(arg.typ, transfer.TransIntegerType):
                        tmp_args = BitVec(arg.name_hint, self.width)
                    else:
                        print(arg.typ)
                        assert False and "not supported type"
                    resultToSMTValue[whole_name] = tmp_args
        else:
            to_smt(op, self.solver, self.funcNameToFunc)


@dataclass
class ToSMTAnalysisPass(ModulePass):
    name = "toSMT"

    def __init__(self, func_name_to_func, solver, width):
        self.solver = solver
        self.funcNameToFunc = func_name_to_func
        self.width = width

    def apply(self, ctx: MLContext, op) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [AssignAttributes(self.funcNameToFunc, self.solver, self.width)]
            ),
            walk_regions_first=False,
            apply_recursively=True,
            walk_reverse=False,
        )
        walker.rewrite_module(op)
