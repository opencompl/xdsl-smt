#!/usr/bin/env python3

import argparse
import itertools
import os
import subprocess as sp
import sys
import time
from multiprocessing import Pool
from typing import Generator, Iterable

from xdsl.context import Context
from xdsl.ir import Attribute
from xdsl.ir.core import BlockArgument, BlockOps, OpResult, SSAValue
from xdsl.parser import Parser

import xdsl_smt.dialects.synth_dialect as synth
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.dialects.builtin import Builtin, ModuleOp, IntegerAttr
from xdsl.dialects.func import Func, FuncOp

from xdsl_smt.cli.xdsl_smt_run import arity, build_interpreter, interpret


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
    arg_parser.add_argument(
        "--summary",
        dest="summary",
        action="store_true",
        help="if present, prints a human-readable summary of the buckets",
    )


MLIR_ENUMERATE = "./mlir-fuzz/build/bin/mlir-enumerate"
SMT_MLIR = "mlir-fuzz/dialects/smt.mlir"


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


def get_inner_func(module: ModuleOp) -> FuncOp:
    assert len(module.ops) == 1
    assert isinstance(module.ops.first, FuncOp)
    return module.ops.first


def func_ops(func: FuncOp) -> BlockOps:
    return func.body.block.ops


def program_size(program: ModuleOp) -> int:
    size = 0
    for op in func_ops(get_inner_func(program)):
        if not isinstance(op, smt.ConstantBoolOp):
            size += 1
    return size


def program_weight(program: ModuleOp) -> tuple[int, int]:
    """
    Returns a tuple to be ordered in lexicographic order.

    First value is the number of non-constant operations in the program. Second
    value is the number of function arguments used.
    """
    return program_size(program), len(
        {
            operand.index
            for op in func_ops(get_inner_func(program))
            for operand in op.operands
            if isinstance(operand, BlockArgument)
        }
    )


def enumerate_programs(
    ctx: Context, max_num_args: int, max_num_ops: int
) -> Generator[ModuleOp, None, None]:
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
        module = Parser(ctx, program).parse_module(True)
        attributes = get_inner_func(module).attributes
        if "seed" in attributes:
            del attributes["seed"]
        yield module
        # Send a character to the enumerator to continue
        assert enumerator.stdin is not None
        enumerator.stdin.write("a")
        enumerator.stdin.flush()


def classify_program(params: tuple[int, ModuleOp]):
    (max_num_args, program) = params
    # Evaluate a program with all possible inputs.
    results: list[bool] = []
    for values in itertools.product((False, True), repeat=max_num_args):
        interpreter = build_interpreter(program, 64)
        arguments: list[Attribute] = [IntegerAttr.from_bool(val) for val in values]
        res = interpret(interpreter, arguments[: arity(interpreter)])
        assert len(res) == 1
        assert isinstance(res[0], bool)
        results.append(res[0])
    return program, tuple(results)


def unify_value(
    lhs_value: SSAValue,
    prog_value: SSAValue,
    left_argument_values: list[SSAValue | None] | None,
) -> bool:
    match lhs_value, prog_value:
        case BlockArgument(index=i), x:
            # If `left_argument_values is None`, then we should match the LHS as
            # is.
            if left_argument_values is None:
                return isinstance(x, BlockArgument) and x.index == i
            # `left_argument_values is not None`: we "unify" the LHS argument
            # with the corresponding program value.
            expected_value = left_argument_values[i]
            if expected_value is None:
                left_argument_values[i] = x
                return True
            return unify_value(expected_value, x, None)
        case OpResult(op=lhs_op, index=i), OpResult(op=prog_op, index=j):
            if i != j:
                return False
            if not isinstance(prog_op, type(lhs_op)):
                return False
            if prog_op.attributes != prog_op.attributes:
                return False
            if len(lhs_op.operands) != len(prog_op.operands):
                return False
            for lhs_operand, prog_operand in zip(
                lhs_op.operands, prog_op.operands, strict=True
            ):
                if not unify_value(lhs_operand, prog_operand, left_argument_values):
                    return False
            return True
        case _:
            return False


def program_is_specialization(program: FuncOp, lhs: FuncOp) -> bool:
    prog_ret = program.get_return_op()
    assert prog_ret is not None

    lhs_ret = lhs.get_return_op()
    assert lhs_ret is not None

    argument_values: list[SSAValue | None] = [None] * len(lhs.args)
    return len(prog_ret.arguments) == len(lhs_ret.arguments) and all(
        map(
            lambda l, p: unify_value(l, p, argument_values),
            lhs_ret.arguments,
            prog_ret.arguments,
        )
    )


def program_contains_lhs(program: FuncOp, lhs: FuncOp) -> bool:
    # We handle this separately because what we do below is testing whether the
    # result of some operation in the program can be expressed in terms of the
    # LHS. Here, we test specifically whether the return values of the program
    # can be expressed in terms of the LHS
    if program_is_specialization(program, lhs):
        return True

    lhs_ret = lhs.get_return_op()
    assert lhs_ret is not None

    for op in func_ops(program):
        argument_values: list[SSAValue | None] = [None] * len(lhs.args)
        if len(op.results) == len(lhs_ret.arguments) and all(
            map(
                lambda l, p: unify_value(l, p, argument_values),
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
            # Don't try to match a program with itself, or with any further
            # program in its bucket.
            if i == program_bucket and k >= program_index:
                break
            # We do not allow matching a program against its canonical
            # representative.
            if k != 0 and program_contains_lhs(
                get_inner_func(program), get_inner_func(lhs)
            ):
                return True
    return False


def remove_superfluous(buckets: Iterable[list[ModuleOp]]) -> int:
    # This defines a total order on the programs of a bucket. The first program
    # of a bucket is called its _canonical representative_, and it is the only
    # program of that bucket that is allowed to appear as a strict subprogram of
    # any program.
    for bucket in buckets:
        bucket.sort(key=program_weight)

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


def pretty_print_value(x: SSAValue, nested: bool = False):
    infix = isinstance(x, OpResult) and (
        isinstance(x.op, smt.BinaryBoolOp)
        or isinstance(x.op, smt.BinaryTOp)
        or isinstance(x.op, smt.IteOp)
    )
    if infix and nested:
        print("(", end="")
    match x:
        case BlockArgument(index=i):
            print(("x", "y", "z", "w", "v", "u", "t", "s")[i], end="")
        case OpResult(op=smt.ConstantBoolOp(value=val), index=0):
            print("⊤" if val else "⊥", end="")
        case OpResult(op=smt.NotOp(arg=arg), index=0):
            print("¬", end="")
            pretty_print_value(arg, True)
        case OpResult(op=smt.AndOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True)
            print(" ∧ ", end="")
            pretty_print_value(rhs, True)
        case OpResult(op=smt.OrOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True)
            print(" ∨ ", end="")
            pretty_print_value(rhs, True)
        case OpResult(op=smt.ImpliesOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True)
            print(" → ", end="")
            pretty_print_value(rhs, True)
        case OpResult(op=smt.DistinctOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True)
            print(" ≠ ", end="")
            pretty_print_value(rhs, True)
        case OpResult(op=smt.EqOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True)
            print(" = ", end="")
            pretty_print_value(rhs, True)
        case OpResult(op=smt.XorOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True)
            print(" ⊕ ", end="")
            pretty_print_value(rhs, True)
        case OpResult(
            op=smt.IteOp(cond=cond, true_val=true_val, false_val=false_val), index=0
        ):
            pretty_print_value(cond, True)
            print(" ? ", end="")
            pretty_print_value(true_val, True)
            print(" : ", end="")
            pretty_print_value(false_val, True)
        case _:
            raise ValueError("Unknown value:", x)
    if infix and nested:
        print(")", end="")


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
    buckets: dict[tuple[bool, ...], list[ModuleOp]] = {
        images: []
        for images in itertools.product((False, True), repeat=2**args.max_num_args)
    }

    start = time.time()

    try:
        print("Counting programs...")
        program_count = get_program_count(args.max_num_args, args.max_num_ops)

        print(f"Classifying {program_count} programs...")
        actual_program_count = 0
        with Pool() as p:
            for i, (program, results) in enumerate(
                p.imap_unordered(
                    classify_program,
                    (
                        (args.max_num_args, program)
                        for program in enumerate_programs(
                            ctx, args.max_num_args, args.max_num_ops
                        )
                    ),
                )
            ):
                buckets[results].append(program)
                percentage = round(i / program_count * 100.0)
                print(f" {i}/{program_count} ({percentage} %)...", end="\r")
                actual_program_count += 1
        print(
            f"Classified {actual_program_count} programs in {int(time.time() - start)} s."
        )

        print("Removing duplicate programs...")
        for i, bucket in enumerate(buckets.values()):
            print(f" {i}/{len(buckets)} buckets done...", end="\r")
            bucket[:] = {str(program): program for program in bucket}.values()
        remaining_count = sum(len(bucket) for bucket in buckets.values())
        print(
            f"Removed {actual_program_count - remaining_count} duplicate programs ({remaining_count} remaining programs)."
        )

        print("Removing superfluous programs...")
        removed_count = remove_superfluous(buckets.values())
        print(f"Removed {removed_count} programs from all buckets")

        remaining_count = sum(len(bucket) for bucket in buckets.values())
        print(f"Remaining programs: {remaining_count}.")

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
                for program in bucket:
                    f.write(str(program))
                    f.write("\n// -----\n")

        if args.summary:
            print("Summary of the generated buckets below:\n")
            for images, bucket in buckets.items():
                for inputs, output in zip(
                    itertools.product((0, 1), repeat=args.max_num_args), images
                ):
                    print(f"\t{inputs} ↦ {int(output)}")
                for program in bucket:
                    ret = get_inner_func(program).get_return_op()
                    assert ret is not None
                    pretty_print_value(ret.arguments[0])
                    print()
                print()

    except BrokenPipeError as e:
        # The enumerator has terminated
        pass
    except Exception as e:
        print(f"Error while enumerating programs: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)


if __name__ == "__main__":
    main()
