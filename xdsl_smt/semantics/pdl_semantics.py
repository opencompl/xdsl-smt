from xdsl_smt.semantics.semantics import OperationSemantics, EffectStates
from typing import Mapping, Sequence, Any
from xdsl.ir import Operation, SSAValue, Attribute
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.interpreters.experimental.pdl import (
    PDLMatcher,
    PDLRewritePattern,
    PDLRewriteFunctions,
)
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
)
from xdsl.context import MLContext
from xdsl.dialects import arith,pdl
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, impl, register_impls
from dataclasses import dataclass, field
from xdsl.utils.exceptions import InterpretationError
from io import StringIO
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect

import sys

@register_impls
@dataclass
class ExtPDLRewriteFunctions(PDLRewriteFunctions):
    
    @impl(pdl.TypeOp)
    def run_type(
        self, interpreter: Interpreter, op: pdl.TypeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        assert isinstance(op.constantType, Attribute)
        return (op.constantType,)

    @impl(pdl.ReplaceOp)
    def run_replace(
        self, interpreter: Interpreter, op: pdl.ReplaceOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        rewriter = self.rewriter

        (old,) = interpreter.get_values((op.op_value,))

        if op.repl_operation is not None:
            (new_op,) = interpreter.get_values((op.repl_operation,))
            rewriter.replace_op(old, new_op)
            self.new_vals = new_op.results
        elif len(op.repl_values):
            new_vals = interpreter.get_values(op.repl_values)
            rewriter.replace_op(old, new_ops=[], new_results=list(new_vals))
            self.new_vals = new_vals
        else:
            assert False, "Unexpected ReplaceOp"

        return ()

class PDLSemantics(OperationSemantics):
    def __init__(self,target_op,rewrite):
        self.target_op = target_op
        self.pdl_rewrite_op = rewrite
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_states: EffectStates,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], EffectStates]:

        # Definitions
        matcher = PDLMatcher()
        pdl_op = self.target_op
        pdl_op_val = pdl_op.results[0]
        xdsl_op = rewriter.current_operation
        parent = self.pdl_rewrite_op.parent_op()
        assert isinstance(parent, pdl.PatternOp)
        pdl_module = parent.parent_op()
        assert isinstance(pdl_module, ModuleOp)
        # Match the source operation.
        assert(matcher.match_operation(pdl_op_val,pdl_op,xdsl_op))
        matched_op = matcher.matching_context[pdl_op_val]
        for constraint_op in parent.walk():
            if isinstance(constraint_op, pdl.ApplyNativeConstraintOp):
                assert(matcher.check_native_constraints(constraint_op))
        # Create the context of the rewriting
        ctx = MLContext()
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(pdl.PDL)
        ctx.load_dialect(SMTDialect)
        ctx.load_dialect(SMTBitVectorDialect)
        # PDLRewriteFUnctions = the RHS pf the rewrite
        functions = ExtPDLRewriteFunctions(ctx)
        functions.rewriter = rewriter
        # The interpreter which performs the actual rewriting
        stream = StringIO()
        interpreter = Interpreter(pdl_module, file=stream)
        interpreter.register_implementations(functions)
        interpreter.push_scope("rewrite")
        interpreter.set_values(matcher.matching_context.items())
        # Go
        interpreter.run_ssacfg_region(self.pdl_rewrite_op.body, ())
        interpreter.pop_scope()
        
        return (functions.new_vals,effect_states)
