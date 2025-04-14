from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.ir import Operation

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects import pdl, arith
from xdsl_smt.semantics.pdl_semantics import PDLSemantics
from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import load_dynamic_semantics
from xdsl_smt.semantics.semantics import OperationSemantics


@dataclass(frozen=True)
class DynamicSemantics(ModulePass):
    """
    Defines the semantics of an operation using the PDL dialect.
    This loads the semantics of the operation in the SMTLowerer, so they can be
    used in the `lower-to-smt` pass.
    """

    name = "dynamic-semantics"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        pdl_module = op
        patterns = [op for op in pdl_module.walk() if isinstance(op, pdl.PatternOp)]
        semantics: dict[type[Operation], OperationSemantics] = {}
        for p in patterns:
            match_ops: list[pdl.OperationOp] = []
            rewrites: list[pdl.RewriteOp] = []
            for pdl_op in p.walk():
                # The matched operation
                if isinstance(pdl_op, pdl.OperationOp) and not isinstance(
                    pdl_op.parent_op(), pdl.RewriteOp
                ):
                    match_ops.append(pdl_op)
                # The SMT-targetting rewrite (semantics)
                elif isinstance(pdl_op, pdl.RewriteOp):
                    is_smt_rewrite = True
                    for inner_op in pdl_op.walk():
                        if isinstance(inner_op, pdl.OperationOp):
                            is_smt_rewrite = is_smt_rewrite and (
                                str(inner_op.opName).split(".")[0] == "smt"
                                # Hacky. Bug in xDSL OperationOp parsing.
                                or str(inner_op.opName).split(".")[0] == '"smt'
                            )
                    assert is_smt_rewrite
                    rewrites.append(pdl_op)
                    break
            assert len(match_ops) == 1
            assert len(rewrites) == 1
            # Agregates the pairs op,semantics
            supported_ops = [arith.AddiOp, arith.ConstantOp]
            for supported_op in supported_ops:
                if (
                    str(match_ops[0].opName) == supported_op.name
                    # Hacky, etc.
                    or str(match_ops[0].opName) == '"' + supported_op.name + '"'
                ):
                    semantics[supported_op] = PDLSemantics(
                        target_op=match_ops[0], rewrite=rewrites[0]
                    )
        # Update the global semantics
        load_dynamic_semantics(semantics)
