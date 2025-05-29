#!/usr/bin/env python3

import sys
import os
import argparse
import subprocess as sp
import time
from typing import Generator

from xdsl.context import Context
from xdsl.parser import Parser

from xdsl_smt.dialects.smt_dialect import BoolAttr
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
import xdsl_smt.dialects.synth_dialect as synth
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl_smt.cli.xdsl_smt_run import interpret_module


MLIR_ENUMERATE = "./mlir-fuzz/build/bin/mlir-enumerate"
SMT_MLIR = "mlir-fuzz/dialects/smt.mlir"


def read_program_from_enumerator(enumerator: sp.Popen[str]) -> str | None:
    program_lines = list[str]()
    assert enumerator.stdout is not None
    while True:
        output = enumerator.stdout.readline()

        # End of program marker
        if output == "// -----\n":
            return "".join(program_lines)

        # End of file
        if not output:
            return None

        # Add the line to the program lines otherwise
        program_lines.append(output)


def enumerate_programs(
    max_num_args: int, max_num_ops: int
) -> Generator[str, None, None]:
    enumerator = sp.Popen(
        [
            MLIR_ENUMERATE,
            SMT_MLIR,
            "--configuration=smt",
            f"--max-num-args={max_num_args}",
            f"--max-num-ops={max_num_ops}",
            "--pause-between-programs",
            "--mlir-print-op-generic",
        ],
        text=True,
        stdin=sp.PIPE,
        stdout=sp.PIPE,
    )

    while (program := read_program_from_enumerator(enumerator)) is not None:
        yield program
        # Send a character to the enumerator to continue
        assert enumerator.stdin is not None
        enumerator.stdin.write("a")
        enumerator.stdin.flush()


def get_program_count(max_num_args: int, max_num_ops: int) -> int:
    return int(
        sp.run(
            [
                MLIR_ENUMERATE,
                SMT_MLIR,
                "--configuration=smt",
                f"--max-num-args={max_num_args}",
                f"--max-num-ops={max_num_ops}",
                "--mlir-print-op-generic",
                "--count",
            ],
            text=True,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
        ).stdout
    )


def all_booleans(count: int) -> Generator[tuple[bool, ...], None, None]:
    assert count >= 0
    if count == 0:
        yield ()
        return
    for tail in all_booleans(count - 1):
        yield (False, *tail)
        yield (True, *tail)


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "--max-num-args",
        type=int,
        help="maximum number of arguments in the generated MLIR programs",
    )
    arg_parser.add_argument(
        "--max-num-ops",
        type=int,
        help="maximum number of operations in the MLIR programs that are generated",
    )
    arg_parser.add_argument(
        "--out-dir",
        type=str,
        help="the directory in which to write the result files",
    )


def main() -> None:
    ctx = Context()
    ctx.allow_unregistered = True
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(SMTDialect)
    ctx.load_dialect(SMTBitVectorDialect)
    ctx.load_dialect(SMTUtilsDialect)
    ctx.load_dialect(synth.SynthDialect)

    # The set of keys is the set of all possible sequences of outputs when
    # presented all the possible inputs. I.e., each key uniquely characterizes a
    # boolean function of arity `args.max_num_args`.
    input_counts = len(list(all_booleans(args.max_num_args)))
    buckets: dict[tuple[bool, ...], list[str]] = {
        images: [] for images in all_booleans(input_counts)
    }

    start = time.time()

    try:
        # Put all programs in buckets.
        program_count = get_program_count(args.max_num_args, args.max_num_ops)
        for i, program in enumerate(
            enumerate_programs(args.max_num_args, args.max_num_ops)
        ):
            percentage = round(i / program_count * 100.0)
            print(f"{i}/{program_count} ({percentage} %)", end="\r")
            module = Parser(ctx, program).parse_module(True)
            # Evaluate a program with all possible inputs.
            results: list[bool] = []
            for values in all_booleans(args.max_num_args):
                res = interpret_module(module, map(BoolAttr, values), 64)
                assert len(res) == 1
                assert isinstance(res[0], bool)
                results.append(res[0])
            buckets[tuple(results)].append(program)

        # Write disk files for each bucket.
        for images, bucket in buckets.items():
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir)
            file_name = "-".join(repr(image) for image in images)
            with open(
                os.path.join(args.out_dir, file_name), "w", encoding="utf-8"
            ) as f:
                for inputs, output in zip(all_booleans(args.max_num_args), images):
                    f.write(f"// {inputs} -> {output}\n")
                f.write("\n")
                for program in bucket:
                    f.write(program)
                    f.write("// -----\n")

        print(f"Classified {program_count} programs in {round(time.time() - start)} s.")

    except BrokenPipeError as e:
        # The enumerator has terminated
        pass
    except Exception as e:
        print(f"Error while enumerating programs: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
