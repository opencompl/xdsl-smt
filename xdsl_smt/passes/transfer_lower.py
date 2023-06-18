from xdsl.dialects.func import *
from xdsl.pattern_rewriter import *
from functools import singledispatch
from dataclasses import dataclass
from xdsl.passes import ModulePass

from xdsl.ir import Operation, MLContext
from ..utils.lower_utils import lowerOperation, CPP_CLASS_KEY, lowerDispatcher

from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
    GreedyRewritePatternApplier,
)
from xdsl.dialects import builtin

autogen = 0


@singledispatch
def transferFunction(op, fout):
    pass


funcStr = ""
indent = "\t"
needDispatch: list[FuncOp] = []


@transferFunction.register
def _(op: Operation, fout):
    global needDispatch
    if isinstance(op, ModuleOp):
        return
        # print(lowerDispatcher(needDispatch))
        # fout.write(lowerDispatcher(needDispatch))
    if len(op.results) > 0 and op.results[0].name_hint is None:
        global autogen
        op.results[0].name_hint = "autogen" + str(autogen)
        autogen += 1
    global funcStr
    if isinstance(op, FuncOp):
        funcDecl = lowerOperation(op)
        funcDecl = funcDecl.format(funcStr)
        funcStr = ""
        # print(funcDecl)
        fout.write(funcDecl)
        if CPP_CLASS_KEY in op.attributes:
            needDispatch.append(op)
    else:
        funcStr += indent + lowerOperation(op)


@dataclass
class LowerOperation(RewritePattern):
    def __init__(self, fout):
        self.fout = fout

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        transferFunction(op, self.fout)


def addDispatcher(fout):
    global needDispatch
    print(lowerDispatcher(needDispatch))
    fout.write(lowerDispatcher(needDispatch))


@dataclass
class LowerToCpp(ModulePass):
    name = "trans_lower"

    def __init__(self, fout):
        self.fout = fout

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerOperation(self.fout)]),
            walk_regions_first=True,
            apply_recursively=True,
            walk_reverse=False,
        )
        walker.rewrite_module(op)
