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

SMTLowerer.rewrite_patterns = {**func_to_smt_patterns, **transfer_to_smt_patterns}
