import xdsl.parser
from xdsl.ir import *
from xdsl.irdl import *
from xdsl.printer import Printer
from xdsl.parser import Parser
from xdsl.dialects.builtin import *
from xdsl.dialects.func import *
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
import dialects.arith_dialect as arith
import dialects.index_dialect as index
from trans_interpreter import getResult

from xdsl.pattern_rewriter import (RewritePattern, PatternRewriter,
                                   op_type_rewrite_pattern,
                                   PatternRewriteWalker,
                                   GreedyRewritePatternApplier)
from xdsl.dialects import mpi, llvm, func, memref, builtin


ATTR_NAME = 'knownBits'


def getBitsAnalysis(op, width):
    if isinstance(op, Operation) and ATTR_NAME in op.attributes:
        return op.attributes[ATTR_NAME]
    return StringAttr('X' * width)


@singledispatch
def transferFunction(op, transferFunctionsMapping:dict):
    pass

@transferFunction.register
def _(op:Operation, transferFunctionsMapping:dict):
    if isinstance(op,FuncOp) or isinstance(op,ModuleOp) or isinstance(op, Return):
        return

    if op.name in transferFunctionsMapping:
        argsDict={}
        width = op.results[0].typ.width.data
        transferFunction=transferFunctionsMapping[op.name]
        for i,arg in enumerate(op.operands):
            knownBits=KnownBits.from_string(getBitsAnalysis(op.operands[i].owner, width).data)
            argsDict[transferFunction.args[i].name]=[knownBits.knownZeros,knownBits.knownOnes]
        result=eval(getResult(transferFunctionsMapping[op.name].get_return_op(),{}),argsDict)
        knownBitsResult=KnownBits(result[0],result[1])
        op.attributes[ATTR_NAME] = StringAttr(knownBitsResult.to_string())
    else:
        print(op)
        assert False and "unsupported op"

@transferFunction.register
def _(op: arith.Constant, transferFunctionsMapping:dict):
    width = op.results[0].typ.width.data
    knownBits= KnownBits.from_constant(width,op.value.value.data)
    op.attributes[ATTR_NAME] = StringAttr(knownBits.to_string())

@dataclass
class AssignAttributes(RewritePattern):

    def __init__(self,transferFunctionsMapping:dict):
        self.transferFunctionsMapping=transferFunctionsMapping

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        transferFunction(op, self.transferFunctionsMapping)

@dataclass
class KnownBitsAnalysisPass(ModulePass):
    name = "kb"

    def __init__(self, transferFunctionsMapping: dict):
        self.transferFunctionsMapping = transferFunctionsMapping

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        walker = PatternRewriteWalker(GreedyRewritePatternApplier([
            AssignAttributes(self.transferFunctionsMapping)
        ]),
            walk_regions_first=True,
            apply_recursively=True,
            walk_reverse=False)
        walker.rewrite_module(op)
