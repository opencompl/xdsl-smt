import argparse
import sys
from pathlib import Path

from xdsl.context import Context
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp
from xdsl.parser import Parser

from xdsl_smt.dialects.llvm_dialect import LLVM
from xdsl_smt.dialects.transfer import Transfer
from xdsl_smt.passes.transfer_lower import LowerToCpp


def _register_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate MLIR code to C++")

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Path to the input MLIR file (defaults to stdin if omitted).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to the output MLIR file (defaults to stdout if omitted).",
    )
    parser.add_argument(
        "--apint", action="store_true", help="Use apints for bitvector type lowering"
    )
    parser.add_argument(
        "--custom_vec",
        action="store_true",
        help="Use custom vec class for transfer value lowering",
    )

    return parser.parse_args()


def _parse_mlir_module(p: Path | None, ctx: Context) -> ModuleOp:
    code = p.read_text() if p else sys.stdin.read()
    fname = p.name if p else "<stdin>"
    mod = Parser(ctx, code, fname).parse_op()

    if isinstance(mod, ModuleOp):
        return mod
    elif isinstance(mod, FuncOp):
        return ModuleOp([mod])
    else:
        raise ValueError(f"mlir in '{fname}' is neither a ModuleOp, nor a FuncOp")


def _get_ctx() -> Context:
    ctx = Context()
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(Transfer)
    ctx.load_dialect(LLVM)

    return ctx


def main() -> None:
    args = _register_args()

    ctx = _get_ctx()
    funcs = _parse_mlir_module(args.input, ctx)
    output = args.output.open("w", encoding="utf-8") if args.output else sys.stdout

    LowerToCpp(
        output, int_to_apint=args.apint, use_custom_vec=args.custom_vec
    ).apply(ctx, funcs)
