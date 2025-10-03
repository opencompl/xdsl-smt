#!/usr/bin/env python3

import sys
import argparse
import re
import os
import subprocess as sp

from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.rewriter import Rewriter

from xdsl_smt.dialects import get_all_dialects
import xdsl_smt.dialects.synth_dialect as synth
from xdsl.dialects.builtin import ModuleOp, IntegerAttr, IntegerType
import xdsl_smt.dialects.hw_dialect as hw


def read_program_from_enumerator(enumerator: sp.Popen[bytes]) -> bytes | None:
    program_lines = list[bytes]()
    assert enumerator.stdout is not None
    while True:
        output = enumerator.stdout.readline()

        # End of program marker
        if output == b"// -----\n":
            return b"".join(program_lines)

        # End of file
        if not output:
            return None

        # Add the line to the program lines otherwise
        program_lines.append(output)


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument("input_file", type=str, help="path to the input file")
    arg_parser.add_argument(
        "--max-num-ops",
        type=int,
        help="number of operations in the MLIR programs that are generated",
    )
    arg_parser.add_argument(
        "--timeout",
        type=int,
        help="The timeout passed to the SMT solver in milliseconds",
        default=8000,
    )
    arg_parser.add_argument(
        "--use-input-ops",
        help="Reuse the existing operations and values",
        action="store_true",
    )
    arg_parser.add_argument(
        "--dialect",
        type=str,
        help="The IRDL file defining the dialect we want to use for synthesis",
    )
    arg_parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        help="Print debugging information in stderr",
        action="store_true",
    )


def replace_synth_with_constants(
    program: ModuleOp, values: list[IntegerAttr[IntegerType]]
) -> None:
    synth_ops: list[synth.ConstantOp] = []
    for op in program.walk():
        if isinstance(op, synth.ConstantOp):
            synth_ops.append(op)

    for op, value in zip(synth_ops, values):
        Rewriter.replace_op(op, hw.ConstantOp(value))


def main() -> None:
    ctx = Context()
    ctx.allow_unregistered = True
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

    with open(args.input_file, "r") as f:
        input_program = Parser(ctx, f.read()).parse_module()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    executable_path = os.path.join(
        current_dir, "..", "..", "mlir-fuzz", "build", "bin", "superoptimizer"
    )

    # Start the enumerator
    enumerator = sp.Popen(
        [
            executable_path,
            args.input_file,
            args.dialect,
            f"--max-num-ops={args.max_num_ops}",
            "--pause-between-programs",
            "--mlir-print-op-generic",
            "--configuration=arith",
            f"--use-input-ops={args.use_input_ops}",
        ],
        stdin=sp.PIPE,
        stdout=sp.PIPE,
    )

    try:
        while True:
            # Read one program from stdin
            program = read_program_from_enumerator(enumerator)

            # End of file
            if program is None:
                break

            # Call the synthesizer with the read program in stdin
            res = sp.run(
                ["xdsl-synth", args.input_file, "-opt"],
                input=program,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
            )
            if res.returncode != 0:
                if args.verbose:
                    print("Example failed:", file=sys.stderr)
                    print(program.decode("utf-8"), file=sys.stderr)
                assert enumerator.stdin is not None
                enumerator.stdin.write(b"a")
                enumerator.stdin.flush()
                continue

            resulting_program = Parser(ctx, res.stdout.decode("utf-8")).parse_module()
            if resulting_program.is_structurally_equivalent(input_program):
                if args.verbose:
                    print("Synthesized the same program:", file=sys.stderr)
                    print(resulting_program, file=sys.stderr)
                assert enumerator.stdin is not None
                enumerator.stdin.write(b"a")
                enumerator.stdin.flush()
                continue

            print(resulting_program.ops.first)
            exit(0)
    except BrokenPipeError as e:
        # The enumerator has terminated
        pass
    except Exception as e:
        print(f"Error while enumerating programs: {e}", file=sys.stderr)
    print("No program found")
    exit(1)


if __name__ == "__main__":
    main()
