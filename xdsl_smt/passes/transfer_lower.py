from dataclasses import dataclass
from typing import TextIO
import sys

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.ir import Operation
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from ..utils.lower_utils import (
    lowerOperation,
    set_use_apint,
    set_use_custom_vec,
    set_use_llvm_kb,
)

autogen = 0
funcStr = ""


def transfer_func(op: Operation, fout: TextIO):
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
    global funcStr
    funcStr += lowerOperation(op)
    parentOp = op.parent_op()
    if isinstance(parentOp, FuncOp) and parentOp.body.block.last_op == op:
        funcStr += "}\n"
        fout.write(funcStr)
        funcStr = ""


@dataclass
class LowerOperation(RewritePattern):
    def __init__(self, fout: TextIO):
        self.fout = fout

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, _: PatternRewriter):
        transfer_func(op, self.fout)


def lower_to_cpp(
    op: ModuleOp,
    fout: TextIO = sys.stdout,
    use_apint: bool = False,
    use_custom_vec: bool = False,
    use_llvm_kb: bool = False,
) -> None:
    global autogen
    autogen = 0

    # set options
    set_use_apint(use_apint)
    set_use_custom_vec(use_custom_vec)
    set_use_llvm_kb(use_llvm_kb)

    PatternRewriteWalker(
        GreedyRewritePatternApplier([LowerOperation(fout)]),
        walk_regions_first=False,
        apply_recursively=False,
        walk_reverse=False,
    ).rewrite_module(op)
