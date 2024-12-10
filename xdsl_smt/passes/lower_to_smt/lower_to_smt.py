"""
Lowers dialects to the SMT dialect
This pass can be extended with additional `RewritePattern`s to
handle more dialects.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.passes import ModulePass
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp, ArrayAttr

from xdsl_smt.passes.lowerers import (
    SMTLowerer,
)


@dataclass(frozen=True)
class LowerToSMTPass(ModulePass):
    name = "lower-to-smt"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        # HACK: This is a temporary solution to get access to the operand types
        # during the lowering.
        # This should be fixed once we have a better lowering scheme that ressemble
        # the conversion patterns in MLIR.
        for sub_op in op.walk():
            sub_op.attributes["__operand_types"] = ArrayAttr(
                [op.type for op in sub_op.operands]
            )
        del op.attributes["__operand_types"]

        lowerer = SMTLowerer()
        lowerer.lower_region(op.body, None)
