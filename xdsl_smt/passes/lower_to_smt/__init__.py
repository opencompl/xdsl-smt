from .lower_to_smt import *

from .func_to_smt import func_to_smt_patterns  # pyright: ignore[reportUnusedImport]
from .llvm_to_smt import llvm_to_smt_patterns  # pyright: ignore[reportUnusedImport]
from .transfer_to_smt import (
    transfer_to_smt_patterns,
)  # pyright: ignore[reportUnusedImport]

SMTLowerer.rewrite_patterns = {**func_to_smt_patterns, **transfer_to_smt_patterns}
