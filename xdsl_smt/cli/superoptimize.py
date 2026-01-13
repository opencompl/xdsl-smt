#!/usr/bin/env python3

import sys
import argparse
import subprocess as sp

from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import load_vanilla_semantics
from xdsl_smt.utils.get_submodule_path import get_mlir_fuzz_executable_path

from xdsl.context import Context
from xdsl.parser import Parser

from xdsl_smt.superoptimization.synthesizer import synthesize_constants
from xdsl_smt.dialects import get_all_dialects
from xdsl.dialects.builtin import ModuleOp


def read_program_from_enumerator(
    enumerator: sp.Popen[bytes], ctx: Context
) -> ModuleOp | None:
    program_lines = list[bytes]()
    assert enumerator.stdout is not None
    while True:
        output = enumerator.stdout.readline()

        # End of program marker
        if output == b"// -----\n":
            return Parser(ctx, b"".join(program_lines).decode("utf-8")).parse_module()

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
        required=True,
    )
    arg_parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        help="Print debugging information in stderr",
        action="store_true",
    )
    arg_parser.add_argument(
        "--configuration",
        dest="configuration",
        type=str,
        help="The configuration to use for synthesis",
        default="arith",
    )
    arg_parser.add_argument(
        "--opt",
        dest="opt",
        help="Optimize SMT queries before sending them to the solver",
        action="store_true",
    )


def main() -> None:
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    ctx = Context()
    ctx.allow_unregistered = True

    load_vanilla_semantics()

    # Register all dialects
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

    with open(args.input_file, "r") as f:
        input_program = Parser(ctx, f.read()).parse_module()

    executable_path = get_mlir_fuzz_executable_path("superoptimizer")

    # Start the enumerator
    enumerator = sp.Popen(
        [
            executable_path,
            args.input_file,
            args.dialect,
            f"--max-num-ops={args.max_num_ops}",
            "--pause-between-programs",
            "--mlir-print-op-generic",
            f"--configuration={args.configuration}",
            f"--use-input-ops={args.use_input_ops}",
        ],
        stdin=sp.PIPE,
        stdout=sp.PIPE,
    )

    try:
        while True:
            # Read one program from stdin
            rhs_program = read_program_from_enumerator(enumerator, ctx)

            # End of file
            if rhs_program is None:
                break

            # Call the synthesizer with the read program in stdin
            result_program = synthesize_constants(
                input_program, rhs_program, ctx, args.opt, args.timeout
            )

            if result_program is None:
                if args.verbose:
                    print("Example failed:", file=sys.stderr)
                    print(rhs_program, file=sys.stderr)
                assert enumerator.stdin is not None
                enumerator.stdin.write(b"a")
                enumerator.stdin.flush()
                continue

            if result_program.is_structurally_equivalent(input_program):
                if args.verbose:
                    print("Synthesized the same program:", file=sys.stderr)
                    print(result_program, file=sys.stderr)
                assert enumerator.stdin is not None
                enumerator.stdin.write(b"a")
                enumerator.stdin.flush()
                continue

            print(result_program.ops.first)
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
