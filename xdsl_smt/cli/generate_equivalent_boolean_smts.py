#!/usr/bin/env python3

import sys
import os
import argparse
import itertools
import subprocess as sp
import time
from multiprocessing import Pool
from typing import Generator, Iterable

from xdsl.context import Context
from xdsl.ir.core import BlockArgument, OpResult, SSAValue
from xdsl.parser import Parser

from xdsl_smt.dialects.smt_dialect import BoolAttr
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
import xdsl_smt.dialects.synth_dialect as synth
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp
from xdsl_smt.cli.xdsl_smt_run import interpret_module


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


def init_ctx():
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
    return args, ctx


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


def classify_program(program: str):
    args, ctx = init_ctx()
    module = Parser(ctx, program).parse_module(True)
    # Evaluate a program with all possible inputs.
    results: list[bool] = []
    for values in itertools.product((False, True), repeat=args.max_num_args):
        res = interpret_module(module, map(BoolAttr, values), 64)
        assert len(res) == 1
        assert isinstance(res[0], bool)
        results.append(res[0])
    return module, tuple(results)


def get_inner_func(module: ModuleOp) -> FuncOp:
    assert len(module.ops) == 1
    assert isinstance(module.ops.first, FuncOp)
    return module.ops.first


def func_ops(func: FuncOp) -> ...:
    assert len(func.body.blocks) == 1
    return func.body.blocks[0].ops


def func_len(func: FuncOp) -> int:
    return len(func_ops(func))


def value_matches(
    argument_values: list[SSAValue | None], lhs_value: SSAValue, prog_value: SSAValue
) -> bool:
    match lhs_value, prog_value:
        case BlockArgument(index=i), x:
            # We "unify" the LHS argument with the corresponding program
            # value.
            if len(argument_values) <= i:
                print(argument_values, lhs_value, i)
            if argument_values[i] is None:
                argument_values[i] = x
                return True
            # TODO: Use `is_structurally_equivalent` here instead? Does it
            # handle block arguments properly? E.g., `%arg0` is not the same as
            # `%arg1`.
            return argument_values[i] == x
        case OpResult(op=lhs_op, index=i), OpResult(op=prog_op, index=j):
            # TODO: Take attributes into account.
            if i != j:
                return False
            if not isinstance(prog_op, type(lhs_op)):
                return False
            if len(lhs_op.operands) != len(prog_op.operands):
                return False
            for lhs_operand, prog_operand in zip(
                lhs_op.operands, prog_op.operands, strict=True
            ):
                if not value_matches(argument_values, lhs_operand, prog_operand):
                    return False
            return True
        case _:
            return False


def program_contains_lhs(program: FuncOp, lhs: FuncOp) -> bool:
    # TODO: Do we want to make this optimization. As long as our programs aren't
    #  guaranteed to be DAGs, I'm not sure.
    # if func_len(lhs) > func_len(program):
    #     return False

    lhs_ret = lhs.get_return_op()
    assert lhs_ret is not None

    prog = program
    assert len(prog.body.blocks) == 1

    for op in prog.body.blocks[0].ops:
        argument_values: list[SSAValue | None] = [None] * len(lhs.args)
        if len(lhs_ret.arguments) == len(op.results) and all(
            map(
                lambda l, p: value_matches(argument_values, l, p),
                lhs_ret.arguments,
                op.results,
            )
        ):
            return True
    return False


def is_program_superfluous(
    buckets: Iterable[list[ModuleOp]],
    program: ModuleOp,
    program_bucket: int,
    program_index: int,
) -> bool:
    for i, bucket in enumerate(buckets):
        for k, lhs in enumerate(bucket):
            # Don't try to find a canonical representative in the program.
            if k == 0:
                continue
            # Don't try to match a program with itself, or with any further
            # program in its bucket.
            if i == program_bucket and k >= program_index:
                break
            if program_contains_lhs(get_inner_func(program), get_inner_func(lhs)):
                return True
    return False


def remove_superfluous(buckets: Iterable[list[ModuleOp]]) -> int:
    # This defines a total order on the programs of a bucket. The first program
    # of a bucket is called its _canonical representative_, and it is the only
    # program of that bucket that is allowed to appear as a strict subprogram of
    # any program.
    for bucket in buckets:
        bucket.sort(key=lambda m: func_len(get_inner_func(m)))

    removed_count = 0
    for i, bucket in enumerate(buckets):
        # We can modify `bucket` inside this loop because we loop over a copy.
        # Since we iterate backwards, we can remove the elements by index.
        for k, program in reversed(list(enumerate(bucket))[1:]):
            if is_program_superfluous(buckets, program, i, k):
                removed_count += 1
                del bucket[k]
            print(
                f"\033[2K Bucket {i}, program {k} (removed {removed_count} in total)",
                end="\r",
            )

    print("\033[2K", end="\r")

    return removed_count


def main() -> None:
    args, _ = init_ctx()

    # The set of keys is the set of all possible sequences of outputs when
    # presented all the possible inputs. I.e., each key uniquely characterizes a
    # boolean function of arity `args.max_num_args`.
    buckets: dict[tuple[bool, ...], list[ModuleOp]] = {
        images: []
        for images in itertools.product((False, True), repeat=2**args.max_num_args)
    }

    start = time.time()

    try:
        print("Counting programs...")
        program_count = get_program_count(args.max_num_args, args.max_num_ops)

        print(f"Classifying {program_count} programs...")
        with Pool() as p:
            for i, (module, results) in enumerate(
                p.imap_unordered(
                    classify_program,
                    enumerate_programs(args.max_num_args, args.max_num_ops),
                )
            ):
                buckets[results].append(module)
                percentage = round(i / program_count * 100.0)
                print(f" {i}/{program_count} ({percentage} %)...", end="\r")
        print(f"Classified {program_count} programs in {int(time.time() - start)} s.")

        # Write disk files for each bucket.
        print("Removing superfluous programs...")
        removed_count = remove_superfluous(buckets.values())
        print(f"Removed {removed_count} programs from all buckets")

        remaining_count = sum(len(bucket) for bucket in buckets.values())
        print(f"Remaining programs: {remaining_count}")

        for images, bucket in buckets.items():
            if not os.path.exists(args.out_dir):
                os.makedirs(args.out_dir)
            file_name = "-".join(repr(image) for image in images)
            with open(
                os.path.join(args.out_dir, file_name), "w", encoding="utf-8"
            ) as f:
                for inputs, output in zip(
                    itertools.product((False, True), repeat=args.max_num_args), images
                ):
                    f.write(f"// {inputs} -> {output}\n")
                f.write("\n")
                for module in bucket:
                    f.write(str(module))
                    f.write("\n// -----\n")

    except BrokenPipeError as e:
        # The enumerator has terminated
        pass
    except Exception as e:
        print(f"Error while enumerating programs: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
