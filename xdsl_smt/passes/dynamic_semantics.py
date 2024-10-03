import sys
from dataclasses import dataclass, field
from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp, IntegerType
from xdsl.dialects import pdl,arith
from xdsl_smt.passes.lower_to_smt.lower_to_smt import SMTLowerer
from xdsl_smt.semantics.pdl_semantics import PDLSemantics

@dataclass(frozen=True)
class DynamicSemantics(ModulePass):
    name = "dynamic-semantics"
    def apply(self, ctx: MLContext, pdl_module: ModuleOp) -> None:
        patterns = [op for op in pdl_module.walk() if isinstance(op, pdl.PatternOp)]
        semantics = {}
        for p in patterns:
            match_ops,rewrites = [],[]
            for op in p.walk():
                # The matched operation
                if (isinstance(op, pdl.OperationOp)
                    and not isinstance(op.parent_op(),pdl.RewriteOp)):
                    match_ops.append(op)
                # The SMT-targetting rewrite (semantics)
                elif isinstance(op, pdl.RewriteOp):
                    is_smt_rewrite = True
                    for inner_op in op.walk():
                        is_smt_rewrite = is_smt_rewrite and (
                            isinstance(inner_op, pdl.OperationOp)
                            and (str(inner_op.opName).split('.')[0] == 'smt'
                                 # Hacky. Bug in xDSL OperationOp parsing.
                                 and str(inner_op.opName).split('.')[0] == '\"smt')
                        )
                    rewrites.append(op)
                    break
            assert(len(match_ops) == 1)
            assert(len(rewrites) == 1)
            # Agregates the pairs op,semantics
            supported_ops = [arith.Addi,arith.Constant]
            for op in supported_ops:
                if (str(match_ops[0].opName) == op.name
                    # Hacky, etc.
                    or str(match_ops[0].opName) == "\"" + op.name + "\""):
                    semantics[op] = PDLSemantics(
                        target_op = match_ops[0],
                        rewrite = rewrites[0]
                    )
        # Update the global semantics
        SMTLowerer.op_semantics = {
            **SMTLowerer.op_semantics,
            **semantics
        }
