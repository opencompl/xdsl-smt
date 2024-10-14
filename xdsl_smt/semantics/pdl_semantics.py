from typing import cast
from xdsl_smt.semantics.semantics import OperationSemantics
from typing import Mapping, Sequence, Any
from xdsl.ir import SSAValue, Attribute, Region, Operation
from xdsl.rewriter import InsertPoint
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.interpreters.experimental.pdl import (
    PDLMatcher,
    PDLRewriteFunctions,
)
from xdsl.context import MLContext
from xdsl.dialects import arith, pdl
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, impl, register_impls
from dataclasses import dataclass
from io import StringIO
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect


def defining_pdl_ops(op: Operation) -> list[pdl.OperationOp]:
    use_chain: list[pdl.OperationOp] = []
    for operand in op.operands:
        if isinstance(operand.owner, pdl.OperationOp) or isinstance(
            operand.owner, pdl.ResultOp
        ):
            use_chain += defining_pdl_ops(operand.owner)
    if isinstance(op, pdl.OperationOp) and isinstance(op.parent_op(), pdl.RewriteOp):
        use_chain.append(op)

    return use_chain


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

        # Insert dependencies if needed
        pdl_ops = defining_pdl_ops(op)
        for pdl_op in pdl_ops:
            if pdl_op.op == op.repl_operation:
                continue
            (new_op,) = interpreter.get_values((pdl_op.op,))
            rewriter.insert_op(new_op, InsertPoint.before(old))

        # Do the replacement itself (and store the new values)
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
    def __init__(self, target_op: pdl.OperationOp, rewrite: pdl.RewriteOp):
        self.target_op = target_op
        self.pdl_rewrite_op = rewrite

    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
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
        assert matcher.match_operation(pdl_op_val, pdl_op, xdsl_op)
        for constraint_op in parent.walk():
            if isinstance(constraint_op, pdl.ApplyNativeConstraintOp):
                assert matcher.check_native_constraints(constraint_op)
        # Create the context of the rewriting
        ctx = MLContext()
        ctx.load_dialect(arith.Arith)
        ctx.load_dialect(pdl.PDL)
        ctx.load_dialect(SMTDialect)
        ctx.load_dialect(SMTUtilsDialect)
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
        interpreter.run_ssacfg_region(cast(Region, self.pdl_rewrite_op.body), ())
        interpreter.pop_scope()

        return (functions.new_vals, effect_state)
