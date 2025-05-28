#!/usr/bin/env python3

import sys
import os
import argparse
import re
import subprocess as sp
from typing import Generator

from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.rewriter import Rewriter

from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl_smt.dialects.hw_dialect import HW
from xdsl_smt.dialects.llvm_dialect import LLVM
import xdsl_smt.dialects.synth_dialect as synth
from xdsl.dialects.builtin import Builtin, ModuleOp, IntegerAttr, IntegerType
from xdsl.dialects.func import Func
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
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


def enumerate_programs(
    max_num_args: int, max_num_ops: int
) -> Generator[bytes, None, None]:
    enumerator = sp.Popen(
        [
            "./mlir-fuzz/build/bin/mlir-enumerate",
            "mlir-fuzz/dialects/smt.mlir",
            "--configuration=smt",
            f"--max-num-args={max_num_args}",
            f"--max-num-ops={max_num_ops}",
            "--pause-between-programs",
            "--mlir-print-op-generic",
        ],
        stdin=sp.PIPE,
        stdout=sp.PIPE,
    )

    while (program := read_program_from_enumerator(enumerator)) is not None:
        yield program
        # Send a character to the enumerator to continue
        assert enumerator.stdin is not None
        enumerator.stdin.write(b"a")
        enumerator.stdin.flush()


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
    buckets = {images: [] for images in all_booleans(input_counts)}

    try:
        # Put all programs in buckets.
        for program in enumerate_programs(args.max_num_args, args.max_num_ops):
            # Evaluate a program with all possible inputs.
            images = []
            for values in all_booleans(args.max_num_args):
                mlir_vals = ",".join(
                    f"#smt.bool_attr<{str(val).lower()}>" for val in values
                )
                res = sp.run(
                    ["xdsl-smt-run", f"--args={mlir_vals}"],
                    input=program,
                    stdout=sp.PIPE,
                    stderr=sp.PIPE,
                )
                # This error case is hit every time we encounter a function with
                # `arity != args.max_num_args`.
                if res.returncode != 0:
                    print(
                        f"Error while evaluating program: {res.stderr.decode('utf-8')}",
                        file=sys.stderr,
                    )
                    print(program.decode("utf8"))
                    break
                if res.stdout.strip() == b"True":
                    images.append(True)
                elif res.stdout.strip() == b"False":
                    images.append(False)
                else:
                    print(
                        "Unexpected output:\n" + res.stdout.decode("utf-8"),
                        file=sys.stderr,
                    )
                    break
            else:
                # The loop exited normally (i.e., no `break`).
                buckets[tuple(images)].append(program)

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
                    f.write(program.decode("utf-8"))
                    f.write("// -----\n")

    except BrokenPipeError as e:
        # The enumerator has terminated
        pass
    except Exception as e:
        print(f"Error while enumerating programs: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
