from .lower_to_smt import *

from .transfer_to_smt import (
    transfer_to_smt_patterns,  # pyright: ignore[reportUnusedImport]
)
from .func_to_smt import func_to_smt_patterns  # pyright: ignore[reportUnusedImport]
from .llvm_to_smt import llvm_to_smt_patterns  # pyright: ignore[reportUnusedImport]

LowerToSMT.rewrite_patterns = [
    *transfer_to_smt_patterns,
    *func_to_smt_patterns,
    *llvm_to_smt_patterns,
]
