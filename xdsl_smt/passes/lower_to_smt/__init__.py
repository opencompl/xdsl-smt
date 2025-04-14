"""
This folder contains the passes to lower MLIR dialects to our semantics dialects.
Each file contains the rewrite patterns or semantics lowerings for a specific
dialect.
smt_lowerer.py and smt_lowerer_patterns.py define the generic infrastructure to
lower the dialects to the semantics dialects.
"""

from .lower_to_smt import *

from .func_to_smt import (
    func_to_smt_patterns as func_to_smt_patterns,
)
from .llvm_to_smt import (
    llvm_to_smt_patterns as llvm_to_smt_patterns,
)
from .transfer_to_smt import (
    transfer_to_smt_patterns as transfer_to_smt_patterns,
)
from xdsl_smt.passes.lower_to_smt.smt_lowerer import (
    SMTLowerer,
)

SMTLowerer.rewrite_patterns = {**func_to_smt_patterns, **transfer_to_smt_patterns}
