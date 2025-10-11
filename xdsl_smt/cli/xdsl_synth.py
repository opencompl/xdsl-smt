#!/usr/bin/env python3

import argparse
import sys

from xdsl.ir import Operation
from xdsl.context import Context
from xdsl.parser import Parser

from xdsl_smt.dialects import get_all_dialects
from xdsl.dialects.builtin import ModuleOp

from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import load_vanilla_semantics
from xdsl_smt.superoptimization.synthesizer import synthesize_constants


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "before_file", type=str, nargs="?", help="path to before input file"
    )

    arg_parser.add_argument(
        "after_file", type=str, nargs="?", help="path to after input file"
    )

    arg_parser.add_argument(
        "-opt",
        help="Optimize the SMTLib program by lowering "
        "pairs and applying constant folding.",
        action="store_true",
    )


def main() -> None:
    ctx = Context()
    ctx.allow_unregistered = True
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

    # Parse the files
    def parse_file(file: str | None) -> Operation:
        if file is None:
            f = sys.stdin
        else:
            f = open(file)

        parser = Parser(ctx, f.read())
        module = parser.parse_module()
        return module

    module = parse_file(args.before_file)
    module_after = parse_file(args.after_file)

    assert isinstance(module, ModuleOp)
    assert isinstance(module_after, ModuleOp)

    load_vanilla_semantics()

    res_rhs = synthesize_constants(module, module_after, ctx, True)
    if res_rhs is None:
        print("Synthesis failed")
        exit(1)
    else:
        print(res_rhs)


if __name__ == "__main__":
    main()
