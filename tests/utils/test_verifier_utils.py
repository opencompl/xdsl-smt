import sys

from xdsl.context import MLContext
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp
from xdsl.ir import Operation
from xdsl.parser import Parser

from xdsl_smt.dialects.index_dialect import Index
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl_smt.dialects.transfer import Transfer
from xdsl_smt.utils.verifier_utils import verify_transfer_function


def parse_file(ctx: MLContext, file: str | None) -> Operation:
    if file is None:
        f = sys.stdin
        file = "<stdin>"
    else:
        f = open(file)

    parser = Parser(ctx, f.read(), file)
    module = parser.parse_op()
    return module


def is_transfer_function(func: FuncOp) -> bool:
    """
    Input: a function with type FuncOp
    Return: True if the function is a transfer function that needs to be verified
            False if the function is a helper function or others
    """

    return "applied_to" in func.attributes


def test_file(file: str) -> bool:
    ctx = MLContext()
    # Register all dialects
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(SMTDialect)
    ctx.load_dialect(SMTBitVectorDialect)
    ctx.load_dialect(SMTUtilsDialect)
    ctx.load_dialect(Transfer)
    ctx.load_dialect(Index)

    # Parse the files
    module = parse_file(ctx, file)
    assert isinstance(module, ModuleOp)
    transfer_function = None
    helper_funcs = []
    func_name_to_func: dict[str, FuncOp] = {}
    for func in module.ops:
        if isinstance(func, FuncOp):
            func_name_to_func[func.sym_name.data] = func
            if is_transfer_function(func):
                transfer_function = func
            else:
                helper_funcs.append(func)
    return verify_transfer_function(transfer_function, helper_funcs, ctx, 8)


assert test_file("./verifier_utils_test/knownBitsAnd.mlir")
assert not test_file("./verifier_utils_test/knownBitsAnd-wrong.mlir")
