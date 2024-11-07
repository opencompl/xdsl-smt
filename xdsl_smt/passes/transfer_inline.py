from dataclasses import dataclass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.ir import Operation

from xdsl_smt.dialects.smt_dialect import CallOp, DefineFunOp, ReturnOp
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp, Call, Return
from xdsl.ir import Operation
from xdsl.context import MLContext
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern


class FunctionCallInlinePattern(RewritePattern):
    def __init__(self, func_name_to_func: dict[str, FuncOp]):
        self.func_name_to_func = func_name_to_func

    @op_type_rewrite_pattern
    def match_and_rewrite(self, callOp: Call, rewriter: PatternRewriter) -> None:
        callee = callOp.callee.string_value()
        value_map: dict[SSAValue, SSAValue] = {}
        func_name_to_func = self.func_name_to_func
        assert callee in func_name_to_func and ("Cannot find the callee " + callee)
        calleeFunc = func_name_to_func[callee]
        for i, arg in enumerate(calleeFunc.args):
            value_map[arg] = callOp.arguments[i]

        for op in calleeFunc.body.ops:
            if isinstance(op, Return):
                callOp.results[0].replace_by(value_map[op.arguments[0]])
                rewriter.erase_matched_op()
                return
            else:
                newOp: Operation = op.clone()
                for i, arg in enumerate(op.operands):
                    newOp.operands[i] = value_map[arg]
                value_map[op.results[0]] = newOp.results[0]
                rewriter.insert_op_before_matched_op(newOp)


class SMTCallInlinePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, callOp: CallOp, rewriter: PatternRewriter) -> None:
        callee = callOp.func.owner
        assert isinstance(callee, DefineFunOp) and "SMT call should call a SMT function"
        value_map: dict[SSAValue, SSAValue] = {}
        for i, arg in enumerate(callOp.args):
            value_map[callee.body.block.args[i]] = arg
        for op in callee.body.ops:
            if isinstance(op, ReturnOp):
                for call_res, ret in zip(callOp.results, op.ret):
                    call_res.replace_by(value_map[ret])
                rewriter.erase_matched_op()
                return
            else:
                newOp = op.clone()
                for i, arg in enumerate(op.operands):
                    newOp.operands[i] = value_map[arg]
                for op_res, newOp_res in zip(op.results, newOp.results):
                    value_map[op_res] = newOp_res
                rewriter.insert_op_before_matched_op(newOp)


@dataclass(frozen=True)
class FunctionCallInline(ModulePass):
    name = "callInline"

    is_SMT_call: bool
    func_name_to_func: dict[str, FuncOp]

    def apply(self, ctx: MLContext, op: ModuleOp):
        if self.is_SMT_call:
            walker = PatternRewriteWalker(SMTCallInlinePattern(), walk_reverse=True)
        else:
            walker = PatternRewriteWalker(
                FunctionCallInlinePattern(self.func_name_to_func), walk_reverse=True
            )
        walker.rewrite_module(op)
