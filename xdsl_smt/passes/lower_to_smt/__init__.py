from .lower_to_smt import *

from .func_to_smt import (
    func_to_smt_patterns as func_to_smt_patterns,
    func_to_smt_with_func_patterns as func_to_smt_with_func_patterns,
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
