#!/usr/bin/env python3

import argparse
import itertools
import os
import subprocess as sp
import sys
import time
from functools import cmp_to_key
from typing import Generator

from xdsl.context import Context
from xdsl.ir import Attribute
from xdsl.ir.core import BlockArgument, BlockOps, Operation, OpResult, SSAValue
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
        "--out-file",
        type=str,
        help="the file in which to write the results",
    )
    arg_parser.add_argument(
        "--summary",
        dest="summary",
        action="store_true",
        help="if present, prints a human-readable summary of the buckets",
    )


MLIR_ENUMERATE = "./mlir-fuzz/build/bin/mlir-enumerate"
SMT_MLIR = "mlir-fuzz/dialects/smt.mlir"
EXCLUDE_SUBPATTERNS_FILE = f"/tmp/exclude-subpatterns-{time.time()}"


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


def formula_size(formula: SSAValue) -> int:
    match formula:
        case BlockArgument():
            return 0
        case OpResult(op=smt.ConstantBoolOp()):
            return 0
        case OpResult(op=op):
            return 1 + sum(formula_size(operand) for operand in op.operands)
        case x:
            raise ValueError(f"Unknown value {x}")


def program_size(program: ModuleOp) -> int:
    ret = get_inner_func(program).get_return_op()
    assert ret is not None
    return sum(formula_size(argument) for argument in ret.arguments)


def enumerate_programs(
    ctx: Context,
    max_num_args: int,
    num_ops: int,
    illegals: list[ModuleOp],
) -> Generator[ModuleOp, None, None]:
    with open(EXCLUDE_SUBPATTERNS_FILE, "w") as f:
        for program in illegals:
            f.write(create_pattern_from_program(program))
            f.write("\n// -----\n")

    enumerator = sp.Popen(
        [
            MLIR_ENUMERATE,
            SMT_MLIR,
            "--configuration=smt",
            f"--max-num-args={max_num_args}",
            f"--max-num-ops={num_ops}",
            "--pause-between-programs",
            "--mlir-print-op-generic",
            f"--exclude-subpatterns={EXCLUDE_SUBPATTERNS_FILE}",
        ],
        text=True,
        stdin=sp.PIPE,
        stdout=sp.PIPE,
    )

    enumerated: set[str] = set()

    while (program := read_program_from_enumerator(enumerator)) is not None:
        # Send a character to the enumerator to continue
        assert enumerator.stdin is not None
        enumerator.stdin.write("a")
        enumerator.stdin.flush()

        module = Parser(ctx, program).parse_module(True)

        if program_size(module) != num_ops:
            continue

        # Deduplication
        attributes = get_inner_func(module).attributes
        if "seed" in attributes:
            del attributes["seed"]
        s = str(module)
        if s in enumerated:
            continue
        enumerated.add(s)

        yield module


def is_same_behavior(max_num_args: int, left: ModuleOp, right: ModuleOp) -> bool:
    # Evaluate programs with all possible inputs.
    left_interpreter = build_interpreter(left, 64)
    right_interpreter = build_interpreter(right, 64)
    for values in itertools.product((False, True), repeat=max_num_args):
        arguments: list[Attribute] = [IntegerAttr.from_bool(val) for val in values]
        left_res = interpret(left_interpreter, arguments[: arity(left_interpreter)])
        right_res = interpret(right_interpreter, arguments[: arity(right_interpreter)])
        if left_res != right_res:
            return False
    return True


def compare_value_lexicographically(left: SSAValue, right: SSAValue) -> int:
    match left, right:
        case BlockArgument(index=i), BlockArgument(index=j):
            return i - j
        case BlockArgument(), OpResult():
            return -1
        case OpResult(), BlockArgument():
            return 1
        # TODO: Consider constant booleans as not equal.
        case OpResult(op=lop, index=i), OpResult(op=rop, index=j):
            if isinstance(lop, type(rop)):
                return i - j
            if len(lop.operands) != len(rop.operands):
                return len(lop.operands) - len(rop.operands)
            for lo, ro in zip(lop.operands, rop.operands, strict=True):
                c = compare_value_lexicographically(lo, ro)
                if c != 0:
                    return c
            return 0
        case l, r:
            raise ValueError(f"Unknown value: {l} or {r}")


def compare_lexicographically(left: ModuleOp, right: ModuleOp) -> int:
    if program_size(left) < program_size(right):
        return -1
    if program_size(right) > program_size(left):
        return 1
    left_ret = get_inner_func(left).get_return_op()
    assert left_ret is not None
    right_ret = get_inner_func(right).get_return_op()
    assert right_ret is not None
    return compare_value_lexicographically(
        left_ret.arguments[0], right_ret.arguments[0]
    )


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
        case OpResult(op=smt.ConstantBoolOp(value=lv)), OpResult(
            op=smt.ConstantBoolOp(value=pv)
        ):
            return lv == pv
        case (OpResult(op=smt.ConstantBoolOp()), _) | (
            _,
            OpResult(op=smt.ConstantBoolOp()),
        ):
            return False
        case OpResult(op=lhs_op, index=i), OpResult(op=prog_op, index=j):
            if i != j:
                return False
            if not isinstance(prog_op, type(lhs_op)):
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


def is_pattern(lhs: ModuleOp, program: ModuleOp) -> bool:
    prog_ret = get_inner_func(program).get_return_op()
    assert prog_ret is not None

    lhs_func = get_inner_func(lhs)
    lhs_ret = lhs_func.get_return_op()
    assert lhs_ret is not None

    argument_values: list[SSAValue | None] = [None] * len(lhs_func.args)
    return len(prog_ret.arguments) == len(lhs_ret.arguments) and all(
        map(
            lambda l, p: unify_value(l, p, argument_values),
            lhs_ret.arguments,
            prog_ret.arguments,
        )
    )


def create_pattern_from_program(program: ModuleOp) -> str:
    lines = [
        "builtin.module {",
        "  pdl.pattern : benefit(1) {",
        # TODO: Support not everything being the same type.
        "    %type = pdl.type",
    ]

    body_start_index = len(lines)

    used_arguments: set[int] = set()
    used_attributes: set[str] = set()
    op_ids: dict[Operation, int] = {}

    i = 0
    for i, op in enumerate(get_inner_func(program).body.ops):
        op_ids[op] = i
        operands: list[str] = []
        for operand in op.operands:
            if isinstance(operand, BlockArgument):
                used_arguments.add(operand.index)
                operands.append(f"%arg{operand.index}")
            elif isinstance(operand, OpResult):
                operands.append(f"%res{op_ids[operand.op]}.{operand.index}")
        ins = (
            f" ({', '.join(operands)} : {', '.join('!pdl.value' for _ in operands)})"
            if len(operands) != 0
            else ""
        )
        attrs = ""
        if isinstance(op, smt.ConstantBoolOp):
            used_attributes.add(str(op.value))
            attrs = f' {{"value" = %attr.{op.value}}}'
        outs = (
            f" -> ({', '.join('%type' for _ in op.results)} : {', '.join('!pdl.type' for _ in op.results)})"
            if len(op.results) != 0
            else ""
        )
        lines.append(f'    %op{i} = pdl.operation "{op.name}"{ins}{attrs}{outs}')
        for j in range(len(op.results)):
            lines.append(f"    %res{i}.{j} = pdl.result {j} of %op{i}")

    lines.append(f'    rewrite %op{i} with "rewriter"')
    lines.append("  }")
    lines.append("}")

    lines[body_start_index:body_start_index] = [
        f"    %arg{k} = pdl.operand" for k in used_arguments
    ]
    lines[body_start_index:body_start_index] = [
        f"    %attr.{attr} = pdl.attribute = {attr}" for attr in used_attributes
    ]

    return "\n".join(lines)


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
        case OpResult(op=smt.AndOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True)
            print(" ∧ ", end="")
            pretty_print_value(rhs, True)
        case OpResult(op=smt.OrOp(operands=(lhs, rhs)), index=0):
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
        case OpResult(op=smt.XOrOp(operands=(lhs, rhs)), index=0):
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


def pretty_print_program(program: ModuleOp):
    ret = get_inner_func(program).get_return_op()
    assert ret is not None
    pretty_print_value(ret.arguments[0])


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

    canonicals: list[ModuleOp] = []
    illegals: list[ModuleOp] = []

    try:
        for m in range(args.max_num_ops + 1):
            print(f"\033[1m== Size {m} ==\033[0m")
            step_start = time.time()

            print("Enumerating programs...")
            new_program_count = 0
            new_illegals: list[ModuleOp] = []
            new_behaviors: list[list[ModuleOp]] = []
            for program in enumerate_programs(ctx, args.max_num_args, m, illegals):
                new_program_count += 1
                print(f" {new_program_count}", end="\r")
                for canonical in canonicals:
                    if is_same_behavior(args.max_num_args, program, canonical):
                        new_illegals.append(program)
                        break
                else:
                    for behavior in new_behaviors:
                        if is_same_behavior(args.max_num_args, program, behavior[0]):
                            behavior.append(program)
                            break
                    else:
                        new_behaviors.append([program])
            print(f"Generated {new_program_count} programs of this size.")
            print(
                f"{sum(len(b) for b in new_behaviors)} of them exhibited {len(new_behaviors)} new behaviors."
            )

            print("Choosing new canonical representatives...")
            for i, behavior in enumerate(new_behaviors):
                print(f" {i + 1}/{len(new_behaviors)}", end="\r")
                # We take the canonical representative to be one that is "as
                # minimal as possible" for the "pattern" preorder relation, in
                # order to be able to add as many programs as possible to
                # `illegals`. This can be approximated by taking the minimum for
                # the lexicographical order.
                canonical = min(behavior, key=cmp_to_key(compare_lexicographically))
                canonicals.append(canonical)
                new_illegals.extend(
                    program
                    for program in behavior
                    if not is_pattern(program, canonical)
                )

            print("Removing redundant illegal subpatterns...")
            size = len(new_illegals)
            redundant_count = 0
            progress = 0
            for i, program in reversed(list(enumerate(new_illegals))):
                progress += 1
                print(f" {progress}/{size}", end="\r")
                if any(
                    j != i and is_pattern(lhs, program)
                    for j, lhs in enumerate(new_illegals)
                ):
                    redundant_count += 1
                    del new_illegals[i]
            illegals.extend(new_illegals)
            print(f"Removed {redundant_count} redundant illegal subpatterns.")

            step_end = time.time()
            print(f"Finished step in {round(step_end - step_start, 2)} s.")

        # Write results to disk.
        old_stdout = sys.stdout
        with open(args.out_file, "w", encoding="UTF-8") as f:
            sys.stdout = f
            for program in canonicals:
                pretty_print_program(program)
                print()
            print()
            for program in illegals:
                pretty_print_program(program)
                print()
        sys.stdout = old_stdout

        if args.summary:
            print(f"\033[1m== Summary (canonical programs) ==\033[0m")
            for program in canonicals:
                pretty_print_program(program)
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
