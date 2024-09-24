#!/usr/bin/env python3

import sys
import argparse
import re
import subprocess as sp

from xdsl.context import MLContext
from xdsl.parser import Parser
from xdsl.rewriter import Rewriter

from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl_smt.dialects.hw_dialect import HW
from xdsl_smt.dialects.llvm_dialect import LLVM
import xdsl_smt.dialects.synth_dialect as smt_synth
from xdsl_smt.dialects.synth_dialect import SMTSynthDialect
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


def replace_synth_with_constants(
    program: ModuleOp, values: list[IntegerAttr[IntegerType]]
) -> None:
    synth_ops: list[smt_synth.ConstantOp] = []
    for op in program.walk():
        if isinstance(op, smt_synth.ConstantOp):
            synth_ops.append(op)

    for op, value in zip(synth_ops, values):
        Rewriter.replace_op(op, hw.ConstantOp(value))


def main() -> None:
    ctx = MLContext()
    ctx.allow_unregistered = True
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(SMTDialect)
    ctx.load_dialect(SMTBitVectorDialect)
    ctx.load_dialect(SMTUtilsDialect)
    ctx.load_dialect(SMTSynthDialect)
    ctx.load_dialect(Comb)
    ctx.load_dialect(HW)
    ctx.load_dialect(LLVM)

    # Start the enumerator
    enumerator = sp.Popen(
        [
            "./mlir-fuzz/build/bin/superoptimizer",
            args.input_file,
            "mlir-fuzz/dialects/arith.mlir",
            f"--max-num-ops={args.max_num_ops}",
            "--pause-between-programs",
            "--mlir-print-op-generic",
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
                print(
                    f"Error while synthesizing program: {res.stderr}", file=sys.stderr
                )
                continue

            res_z3 = sp.run(
                ["z3", "-in", f"-T:{args.timeout}"],
                input=res.stdout + b"\n(get-model)",
                stdout=sp.PIPE,
                stderr=sp.PIPE,
            )

            if "model is not available" not in res_z3.stdout.decode():
                values_str: list[str] = re.findall(
                    r"#([xb][0-9a-f]+)", res_z3.stdout.decode()
                )
                values: list[IntegerAttr[IntegerType]] = []
                for value in values_str:
                    if value.startswith("x"):
                        val = int(value[1:], 16)
                        bitwidth = len(value[1:]) * 4
                    else:
                        val = int(value[1:], 2)
                        bitwidth = len(value[1:])
                    values.append(IntegerAttr(val, bitwidth))

                mlir_program = Parser(ctx, program.decode()).parse_module()
                replace_synth_with_constants(mlir_program, values)

                print(mlir_program)

            # Set a character to the enumerator to continue
            assert enumerator.stdin is not None
            enumerator.stdin.write(b"a")
            enumerator.stdin.flush()
    except BrokenPipeError as e:
        # The enumerator has terminated
        pass
    except Exception as e:
        print(f"Error while enumerating programs: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
