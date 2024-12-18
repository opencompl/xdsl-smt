from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl_smt.semantics.generic_integer_proxy import IntegerProxy
from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import load_int_semantics
from xdsl.pattern_rewriter import PatternRewriter


@dataclass(frozen=True)
class LoadIntSemanticsPass(ModulePass):
    name = "load-int-semantics"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        if op.body.first_block is None:
            return
        assert op.body.first_block
        if op.body.first_block.first_op:
            return
        assert op.body.first_block.first_op
        rewriter = PatternRewriter(op.body.first_block.first_op)
        integer_proxy = IntegerProxy()
        integer_proxy.build_pow2(rewriter)
        load_int_semantics(integer_proxy)
