import xdsl.parser
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.printer import Printer
from xdsl.parser import Parser
from xdsl.dialects.builtin import *
import dialects.arith_dialect as arith
import dialects.index_dialect as index
from xdsl.dialects.transfer import AbstractValueType, IfOp, GetOp, MakeOp, NegOp
from xdsl.dialects.func import FuncOp, Return
from xdsl.dialects.scf import *
from xdsl.pattern_rewriter import *
from functools import singledispatch
from abc import ABC
from typing import TypeVar, cast
from dataclasses import dataclass
from xdsl.utils.knownBits import *
from xdsl.passes import ModulePass


from xdsl.utils.hints import isa
from xdsl.dialects.builtin import Signedness, IntegerType, i32, i64, IndexType
from xdsl.dialects.memref import MemRefType
from xdsl.ir import Operation, SSAValue, OpResult, Attribute, MLContext

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.dialects import mpi, llvm, func, memref, builtin

#assume we already did renaming process
@singledispatch
def getResult(op, funcDict: dict):
    if isinstance(op,BlockArgument):
        return op.name
    print(op)
    assert False and "unsupported op"

@getResult.register
def _(op: arith.Addi, funcDict: dict):
    if op.results[0].name not in funcDict:
        lResult=getResult(op.lhs, funcDict)
        rResult=getResult(op.rhs, funcDict)
        result=lResult+"+"+rResult
        funcDict[op.results[0].name]=result
    return funcDict[op.results[0].name]

@getResult.register
def _(op:OpResult, funcDict:dict):
    return getResult(op.op,funcDict)


@getResult.register
def _(op: arith.Constant, funcDict: dict):
    return str(op.value.value.data)

@getResult.register
def _(op: index.Constant, funcDict: dict):
    return str(op.value.value.data)


@getResult.register
def _(op: arith.Andi, funcDict: dict):
    if op.results[0].name not in funcDict:
        lResult=getResult(op.lhs, funcDict)
        rResult=getResult(op.rhs, funcDict)
        result=lResult+"&"+rResult
        funcDict[op.results[0].name]=result
    return funcDict[op.results[0].name]

@getResult.register
def _(op: arith.Ori, funcDict: dict):
    if op.results[0].name not in funcDict:
        lResult=getResult(op.lhs, funcDict)
        rResult=getResult(op.rhs, funcDict)
        result=lResult+"|"+rResult
        funcDict[op.results[0].name]=result
    return funcDict[op.results[0].name]

@getResult.register
def _(op: MakeOp, funcDict: dict):
    if op.results[0].name not in funcDict:
        args="["
        for arg in op.arguments:
            args+=getResult(arg.owner,funcDict)
            args+=','
        result=args+"]"

        funcDict[op.results[0].name]=result
    return funcDict[op.results[0].name]

@getResult.register
def _(op: GetOp, funcDict: dict):
    if op.results[0].name not in funcDict:
        absVal=getResult(op.absVal, funcDict)
        index=getResult(op.index, funcDict)
        result=absVal+"["+index+"]"
        funcDict[op.results[0].name]=result
    return funcDict[op.results[0].name]

@getResult.register
def _(op: Return, funcDict: dict):
    return getResult(op.arguments[0], funcDict)