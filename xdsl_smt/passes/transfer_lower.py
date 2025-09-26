from dataclasses import dataclass
from functools import singledispatch
from typing import TextIO

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from ..utils.lower_utils import (
    CPP_CLASS_KEY,
    INDUCTION_KEY,
    lowerDispatcher,
    lowerInductionOps,
    lowerOperation,
    set_int_to_apint,
)

autogen = 0


@singledispatch
def transferFunction(op: Operation, fout: TextIO):
    pass


indent = "\t"
funcPrefix = 'extern "C" '
funcStr = funcPrefix
needDispatch: list[FuncOp] = []
inductionOp: list[FuncOp] = []


@transferFunction.register
def _(op: Operation, fout: TextIO):
    global needDispatch
    global inductionOp
    if isinstance(op, ModuleOp):
        return
    if len(op.results) > 0 and op.results[0].name_hint is None:
        global autogen
        op.results[0].name_hint = "autogen" + str(autogen)
        autogen += 1
    if isinstance(op, FuncOp):
        for arg in op.args:
            if arg.name_hint is None:
                arg.name_hint = "autogen" + str(autogen)
                autogen += 1
        if CPP_CLASS_KEY in op.attributes:
            needDispatch.append(op)
        if INDUCTION_KEY in op.attributes:
            inductionOp.append(op)
    global funcStr
    funcStr += lowerOperation(op)
    parentOp = op.parent_op()
    if isinstance(parentOp, FuncOp) and parentOp.body.block.last_op == op:
        funcStr += "}\n"
        fout.write(funcStr)
        funcStr = funcPrefix


@dataclass
class LowerOperation(RewritePattern):
    def __init__(self, fout: TextIO):
        self.fout = fout

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, _: PatternRewriter):
        transferFunction(op, self.fout)


def addInductionOps(fout: TextIO):
    global inductionOp
    if len(inductionOp) != 0:
        fout.write(lowerInductionOps(inductionOp))


def addDispatcher(fout: TextIO, is_forward: bool):
    global needDispatch
    if len(needDispatch) != 0:
        fout.write(lowerDispatcher(needDispatch, is_forward))


@dataclass(frozen=True)
class LowerToCpp(ModulePass):
    name = "trans_lower"
    fout: TextIO
    int_to_apint: bool = False

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        global autogen
        autogen = 0
        set_int_to_apint(self.int_to_apint)
        # We found PatternRewriteWalker skipped the op itself during iteration
        # Do it manually on op
        transferFunction(op, self.fout)
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerOperation(self.fout)]),
            walk_regions_first=False,
            apply_recursively=False,
            walk_reverse=False,
        )
        walker.rewrite_module(op)
