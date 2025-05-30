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
from xdsl.ir import Operation
from xdsl.ir.core import BlockArgument, OpResult, SSAValue
from xdsl.parser import Parser
from xdsl.pattern_rewriter import (
    op_type_rewrite_pattern,
    PatternRewriter,
    RewritePattern,
)

from xdsl_smt.dialects.smt_dialect import BoolAttr
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
import xdsl_smt.dialects.synth_dialect as synth
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp, ReturnOp
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


def value_matches(lhs_value: SSAValue, prog_value: SSAValue) -> bool:
    match lhs_value, prog_value:
        case BlockArgument(index=i), BlockArgument(index=j):
            return i == j
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
                if not value_matches(lhs_operand, prog_operand):
                    return False
            return True
        case _:
            return False


class PeepholeRewriter(RewritePattern):
    lhs_ret: ReturnOp
    rhs_ops: list[Operation]
    rhs_res: tuple[SSAValue, ...]

    def __init__(self, lhs: FuncOp, rhs: FuncOp):
        assert len(lhs.body.blocks) == 1
        assert len(rhs.body.blocks) == 1

        lhs_ret = lhs.get_return_op()
        assert lhs_ret is not None
        self.lhs_ret = lhs_ret

        rhs_ret = rhs.get_return_op()
        assert rhs_ret is not None
        # The last operation is a return.
        self.rhs_ops = list(rhs.body.blocks[0].ops)[:-1]
        self.rhs_res = rhs_ret.arguments

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        # We assume operands have no side-effect.
        if len(self.lhs_ret.arguments) != len(op.results):
            return
        for lhs_op, prog_op in zip(self.lhs_ret.arguments, op.results, strict=True):
            if not value_matches(lhs_op, prog_op):
                return
        # Add all operations from the LHS
        # TODO: Make sure `self.rhs_res` still has a meaning when it's used in
        #  another block.
        rewriter.replace_matched_op([op.clone() for op in self.rhs_ops], self.rhs_res)


def try_rewrite(lhs: ModuleOp, rhs: ModuleOp, program: ModuleOp) -> ModuleOp | None:
    new_program = program.clone()
    for op in func_ops(get_inner_func(new_program)):
        rewriter = PatternRewriter(op)
        PeepholeRewriter(get_inner_func(lhs), get_inner_func(rhs)).match_and_rewrite(
            op, rewriter
        )
        if rewriter.has_done_action:
            # print(get_inner_func(lhs))
            # print(get_inner_func(rhs))
            # print(get_inner_func(program))
            # print(get_inner_func(new_program))
            # print()
            return new_program
    return None


def remove_superfluous(bucket: list[ModuleOp]) -> int:
    # Programs appear in the bucket by increasing number of operations. The
    # exact order of programs in the bucket is used when arbitrary decisions
    # have to be made. This arbitrary total strict order on programs is denoted
    # by <. It is a superset of the strict order on program induced by their
    # sizes.
    bucket.sort(key=lambda m: func_len(get_inner_func(m)))

    # Remove any program that can be rewriten into another program of the bucket
    # using a single rewrite rule that arises from other programs of the bucket.
    removed_count = 0
    removed = [False] * len(bucket)
    for k, q1m in enumerate(bucket):
        print(f" {k}/{len(bucket)}, removed {removed_count}", end="\r")
        q1 = get_inner_func(q1m)
        for i, p1m in enumerate(bucket):
            p1 = get_inner_func(p1m)
            if func_len(p1) >= func_len(q1):
                break
            # |p1| < |q1|
            for p2m in bucket[:i]:
                # p2 < p1
                q2m = try_rewrite(p1m, p2m, q1m)
                if q2m is None:
                    continue
                # q1 ~>_p1^p2 q2
                if len(list(filter(q2m.is_structurally_equivalent, bucket))) != 0:
                    continue
                # q2 ∈ initial_bucket
                q2 = get_inner_func(q2m)
                # Should be the case because p2 < p1, meaning |p2| <= |p1| and
                # therefore |q2| <= |q1|.
                assert func_len(q2) <= func_len(q1)
                # |q2| <= |q1|
                if removed[k]:
                    continue
                removed[k] = True
                removed_count += 1
                # If |q1| = |q2|, is it possible that we later discover that
                # q2 ~>_p'1^p'2 q1 for some p'2 < p'1, and thus remove q2 as
                # well?
                # No. We must have (p'1, p'2) = (p2, p1), and thus p'1 < p'2,
                # which prevents us from getting to this point.

    # Mutate the argument
    new_bucket = [module for module, remove in zip(bucket, removed) if not remove]
    del bucket[:]
    bucket.extend(new_bucket)

    return removed_count


def matches(lhs: ModuleOp, program: ModuleOp) -> bool:
    # We assume operands have no side-effect.

    lhs_ret = get_inner_func(lhs).get_return_op()
    assert lhs_ret is not None

    prog = get_inner_func(program)
    assert len(prog.body.blocks) == 1

    for op in prog.body.blocks[0].ops:
        if len(lhs_ret.arguments) == len(op.results) and all(
            map(lambda l, p: value_matches(l, p), lhs_ret.arguments, op.results)
        ):
            return True
    return False


def remove_superfluous_2(buckets: Iterable[list[ModuleOp]]) -> int:
    # We assume all buckets are sorted by program size (this is done by
    # `remove_superfluous`). The exact order of programs in buckets is used when
    # arbitrary decisions have to be made. This arbitrary total strict order on
    # programs of a bucket is denoted by <. It is a superset of the strict order
    # on programs induced by their sizes. The _canonical representant_ of a
    # bucket is the first program of that bucket. It is guaranteed to be present
    # in the superfluous-less bucket.

    removed_count = 0

    for i, program_bucket in enumerate(buckets):
        remove_indices: list[int] = []
        for k, program in enumerate(program_bucket):
            # If we can match the program with any non-canonical representant of
            # a different bucket, we remove the program.
            # This is correct because the matched sub-program can be rewriten to
            # the canonical representant of the LHS's bucket to create a new
            # program P' whose size does not exceed that of `program`.
            # Therefore, P' was enumerated. Now, suppose we remove P' from this
            # bucket. This has to be because it matched an LHS. This LHS cannot
            # be the canonical representant of a bucket, so there exists a
            # program P'' in this bucket with P'' != P' and P'' != `program`
            # that can be obtained from `program` by applying rewrite rules.
            for j, lhs_bucket in enumerate(buckets):
                if i == j:
                    continue
                if any(map(lambda lhs: matches(lhs, program), lhs_bucket)):
                    remove_indices.append(k)
                    removed_count += 1
                    break
        for l in reversed(remove_indices):
            del program_bucket[l]

    return removed_count


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
        for images, bucket in buckets.items():
            initial_bucket_size = len(bucket)
            removed_count = remove_superfluous(bucket)
            print(
                f"Removed {removed_count}/{initial_bucket_size} programs from bucket {images}"
            )
        additional_removal_count = remove_superfluous_2(buckets.values())
        print(
            f"Removed {additional_removal_count} additional programs from all buckets"
        )

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
                    f.write("// -----\n")

    except BrokenPipeError as e:
        # The enumerator has terminated
        pass
    except Exception as e:
        print(f"Error while enumerating programs: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
