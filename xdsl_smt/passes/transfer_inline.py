from xdsl.irdl import IRDLOperation
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.ir import Operation

from ..dialects import smt_bitvector_dialect as smt_bv
from ..dialects import smt_dialect as smt
from ..dialects import transfer
from xdsl.ir import Attribute
from .lower_to_smt import LowerToSMT
from ..dialects.smt_utils_dialect import PairType, SecondOp, FirstOp, PairOp
from xdsl_smt.dialects.smt_dialect import BoolType, CallOp, DefineFunOp, ReturnOp
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp, Call, Return
from xdsl.ir import MLContext, Operation
from xdsl.ir import Operation, Region, SSAValue, Attribute
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
                callOp.results[0].replace_by(value_map[op.ret])
                rewriter.erase_matched_op()
            else:
                newOp = op.clone()
                for i, arg in enumerate(op.operands):
                    newOp.operands[i] = value_map[arg]
                if len(op.results) != 0:
                    value_map[op.results[0]] = newOp.results[0]
                rewriter.insert_op_before_matched_op(newOp)


class FunctionCallInline(ModulePass):
    name = "callInline"

    def __init__(self, is_SMT_call, func_name_to_func: dict[str, FuncOp]):
        self.func_name_to_func = func_name_to_func
        self.is_SMT_call = is_SMT_call

    def apply(self, ctx: MLContext, op: ModuleOp):
        if self.is_SMT_call:
            walker = PatternRewriteWalker(SMTCallInlinePattern(), walk_reverse=True)
        else:
            walker = PatternRewriteWalker(
                FunctionCallInlinePattern(self.func_name_to_func), walk_reverse=True
            )
        walker.rewrite_module(op)
