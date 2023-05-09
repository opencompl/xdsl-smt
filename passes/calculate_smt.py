from xdsl.ir import Attribute, MLContext
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp, FunctionType
from xdsl.dialects.func import FuncOp, Return
from xdsl.dialects.transfer import AbstractValueType, IfOp,GetOp,MakeOp
from xdsl.ir import OpResult
from xdsl.passes import ModulePass
from xdsl.utils.hints import isa
from z3 import *

import dialects.smt_bitvector_dialect as bv_dialect
import dialects.arith_dialect as arith
from dialects.smt_bitvector_dialect import BitVectorType
from dialects.smt_dialect import DefineFunOp, ReturnOp

WIDTH=32

def convert_type(type: Attribute) -> Attribute:
    """Convert a type to an SMT sort"""
    if isinstance(type, IntegerType):
        return BitVectorType(type.width)
    return type
    raise Exception("Cannot convert {type} attribute")

def setSMTVal(op,smtVal):
    setattr(op,"smtVal",smtVal)

def getSMTVal(op):
    if isinstance(op,OpResult):
        op=op.op
    if not hasattr(op,"smtVal"):
        print(op)
        assert False and "the operation doesn't contain smtVal"
    return getattr(op,"smtVal")

class IntegerConstantRewritePattern(RewritePattern):
    nameCnt:int=0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter):
        if not isa(op.value, IntegerAttr[IntegerType]):
            raise Exception("Cannot convert constant of type that are not integer type")
        if op.results[0].name is None:
            op.results[0].name = "autogen_constant"+str(IntegerConstantRewritePattern.nameCnt)
            IntegerConstantRewritePattern.nameCnt+=1
        setSMTVal(op,BitVecVal(op.value.value.data,WIDTH))

class AddiRewritePattern(RewritePattern):
    nameCnt:int = 0
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Addi, rewriter: PatternRewriter):
        if op.results[0].name is None:
            op.results[0].name = "auto_genaddi"+str(AddiRewritePattern.nameCnt)
            AddiRewritePattern.nameCnt+=1
        setSMTVal(op,getSMTVal(op.operands[0])+getSMTVal(op.operands[1]))

class AndiRewritePattern(RewritePattern):
    nameCnt:int = 0
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Andi, rewriter: PatternRewriter):
        if op.results[0].name is None:
            op.results[0].name = "auto_genandi"+str(AndiRewritePattern.nameCnt)
            AndiRewritePattern.nameCnt+=1
        setSMTVal(op,getSMTVal(op.operands[0])&getSMTVal(op.operands[1]))

class OriRewritePattern(RewritePattern):
    nameCnt:int=0
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Ori, rewriter: PatternRewriter):
        if op.results[0].name is None:
            op.results[0].name = "auto_genori" + str(OriRewritePattern.nameCnt)
            OriRewritePattern.nameCnt += 1
        setSMTVal(op, getSMTVal(op.operands[0]) | getSMTVal(op.operands[1]))

class ReturnPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Return, rewriter: PatternRewriter):
        if len(op.arguments) != 1:
            raise Exception("Cannot convert functions with multiple results")



class TransferMakePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MakeOp, rewriter: PatternRewriter):
        smtVals=[]
        for op in op.operands:
            smtVals.append(getSMTVal(op))
        setSMTVal(op,smtVals)

class TransferGetPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: GetOp, rewriter: PatternRewriter):
        smtVals=getSMTVal(op.operands[0])
        index=op.operands[1].op.value.value.data
        setSMTVal(op,smtVals[index])

class FuncToSMTPattern(RewritePattern):
    """Convert func.func to an SMT formula"""
    nameCnt:int=0
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter):
        """
        Convert a `func` function to an smt function.
        """
        # We only handle single-block regions for now
        if len(op.body.blocks) != 1:
            raise Exception("Cannot convert multi-block functions")
        if len(op.function_type.outputs.data) != 1:
            raise Exception("Cannot convert functions with multiple results")


        if not hasattr(op, "smtArgs"):
            setattr(op, "smtArgs", [None for i in range(len(op.args))])
        smtArgs = getattr(op, "smtArgs")
        for i ,arg in enumerate(op.body.blocks[0].args):
            if arg.name is None:
                arg.name="autogen_arg"+str(FuncToSMTPattern.nameCnt)
                FuncToSMTPattern.nameCnt+=1
            if smtArgs[i] is None:
                tmpArgs=None
                if isinstance(arg.typ,AbstractValueType):
                    tmpArgs=[]
                    for j in range(arg.typ.get_num_fields()):
                        tmpArgs.append(BitVec(arg.name+"field"+str(j),WIDTH))
                elif isinstance(arg.typ,IntegerType):
                    tmpArgs=BitVec(arg.name,arg.typ.width.data)
                else:
                    print(arg.typ)
                    assert False and "not supported type"
                smtArgs[i]=tmpArgs
            setSMTVal(arg,smtArgs[i])
        setattr(op,"smtArgs",[])

class CalculateSMT(ModulePass):
    name = "calc-smt"

    def apply(self, ctx: MLContext, op):
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    IntegerConstantRewritePattern(),
                    AddiRewritePattern(),
                    OriRewritePattern(),
                    AndiRewritePattern(),
                    FuncToSMTPattern(),
                    TransferGetPattern(),
                    TransferMakePattern(),
                    ReturnPattern(),
                ]
            )
        )
        walker.rewrite_module(op)
        print(op.sym_name)
        print("finished")
