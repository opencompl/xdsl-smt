#!/usr/bin/env python3

import argparse
import itertools
import subprocess as sp
import sys
import time
import z3  # pyright: ignore[reportMissingTypeStubs]
from io import StringIO
from typing import Any, Generator, IO, Iterable, Sequence, TypeVar, cast

from xdsl.context import Context
from xdsl.ir.core import BlockArgument, OpResult, SSAValue
from xdsl.parser import Parser
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.builder import Builder

import xdsl_smt.dialects.synth_dialect as synth
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_bitvector_dialect as bv
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp, ReturnOp

from xdsl_smt.traits.smt_printer import print_to_smtlib


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
        "--bitvector-widths",
        type=str,
        help="a list of comma-separated bitwidths",
        default="4",
    )


MLIR_ENUMERATE = "./mlir-fuzz/build/bin/mlir-enumerate"
REMOVE_REDUNDANT_PATTERNS = "./mlir-fuzz/build/bin/remove-redundant-patterns"
SMT_MLIR = "./mlir-fuzz/dialects/smt.mlir"
PROGRAM_OUTPUT = f"/tmp/sanity-checker{time.time()}"


@staticmethod
def pretty_print_value(x: SSAValue, nested: bool, *, file: IO[str] = sys.stdout):
    infix = isinstance(x, OpResult) and len(x.op.operand_types) > 1
    parenthesized = infix and nested
    if parenthesized:
        print("(", end="", file=file)
    match x:
        case BlockArgument(index=i, type=smt.BoolType()):
            print(("x", "y", "z", "w", "v", "u", "t", "s")[i], end="", file=file)
        case BlockArgument(index=i, type=bv.BitVectorType(width=width)):
            print(("x", "y", "z", "w", "v", "u", "t", "s")[i], end="", file=file)
            print(f"#{width.data}", end="", file=file)
        case OpResult(op=smt.ConstantBoolOp(value=val), index=0):
            print("⊤" if val else "⊥", end="", file=file)
        case OpResult(op=bv.ConstantOp(value=val), index=0):
            width = val.type.width.data
            value = val.value.data
            print(f"{{:0{width}b}}".format(value), end="", file=file)
        case OpResult(op=smt.NotOp(arg=arg), index=0):
            print("¬", end="", file=file)
            pretty_print_value(arg, True, file=file)
        case OpResult(op=smt.AndOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, file=file)
            print(" ∧ ", end="", file=file)
            pretty_print_value(rhs, True, file=file)
        case OpResult(op=smt.OrOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, file=file)
            print(" ∨ ", end="", file=file)
            pretty_print_value(rhs, True, file=file)
        case OpResult(op=smt.ImpliesOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True, file=file)
            print(" → ", end="", file=file)
            pretty_print_value(rhs, True, file=file)
        case OpResult(op=smt.DistinctOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True, file=file)
            print(" ≠ ", end="", file=file)
            pretty_print_value(rhs, True, file=file)
        case OpResult(op=smt.EqOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True, file=file)
            print(" = ", end="", file=file)
            pretty_print_value(rhs, True, file=file)
        case OpResult(op=smt.XOrOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, file=file)
            print(" ⊕ ", end="", file=file)
            pretty_print_value(rhs, True, file=file)
        case OpResult(
            op=smt.IteOp(cond=cond, true_val=true_val, false_val=false_val), index=0
        ):
            pretty_print_value(cond, True, file=file)
            print(" ? ", end="", file=file)
            pretty_print_value(true_val, True, file=file)
            print(" : ", end="", file=file)
            pretty_print_value(false_val, True, file=file)
        case OpResult(op=bv.AddOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, file=file)
            print(" + ", end="", file=file)
            pretty_print_value(rhs, True, file=file)
        case OpResult(op=bv.AndOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, file=file)
            print(" & ", end="", file=file)
            pretty_print_value(rhs, True, file=file)
        case OpResult(op=bv.OrOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, file=file)
            print(" | ", end="", file=file)
            pretty_print_value(rhs, True, file=file)
        case OpResult(op=bv.MulOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True)
            print(" * ", end="", file=file)
            pretty_print_value(rhs, True, file=file)
        case OpResult(op=bv.NotOp(arg=arg), index=0):
            print("~", end="", file=file)
            pretty_print_value(arg, True, file=file)
        case _:
            raise ValueError(f"Unknown value for pretty print: {x}")
    if parenthesized:
        print(")", end="", file=file)


def pretty_print_func(
    func: smt.DefineFunOp,
    *,
    end: str = "\n",
    file: IO[str] = sys.stdout,
):
    assert func.body.ops.last is not None
    assert len(func.body.ops.last.operands) == 1
    pretty_print_value(func.body.ops.last.operands[0], False, file=file)
    print(end=end, file=file)


def func_to_string(func: smt.DefineFunOp) -> str:
    buffer = StringIO()
    pretty_print_func(func, end="", file=buffer)
    return buffer.getvalue()


def clone_func_to_smt_func(func: FuncOp) -> smt.DefineFunOp:
    """
    Convert a `func.func` to an `smt.define_fun` operation.
    Do not mutate the original function.
    """
    new_region = func.body.clone()

    # Replace the `func.return` with an `smt.return` operation.
    func_return = new_region.block.last_op
    assert isinstance(func_return, ReturnOp)
    rewriter = Rewriter()
    rewriter.insert_op(
        smt.ReturnOp(func_return.arguments), InsertPoint.before(func_return)
    )
    rewriter.erase_op(func_return)

    return smt.DefineFunOp(new_region)


def read_program_from_lines(ctx: Context, lines: IO[str]) -> smt.DefineFunOp | None:
    program_lines = list[str]()
    while True:
        output = lines.readline()

        # End of program marker
        if output == "// -----\n":
            source = "".join(program_lines)
            module = Parser(ctx, source).parse_module(True)
            assert len(module.ops) == 1
            assert isinstance(module.ops.first, FuncOp)
            return clone_func_to_smt_func(module.ops.first)

        # End of file
        if not output:
            return None

        # Add the line to the program lines otherwise
        program_lines.append(output)


def enumerate_programs(
    ctx: Context,
    max_num_args: int,
    max_num_ops: int,
    bitvector_widths: str,
) -> Generator[smt.DefineFunOp, None, None]:
    enumerator = sp.Popen(
        [
            MLIR_ENUMERATE,
            SMT_MLIR,
            "--configuration=smt",
            f"--smt-bitvector-widths={bitvector_widths}",
            "--cse",
            "--seed=1",
            f"--max-num-args={max_num_args}",
            f"--max-num-ops={max_num_ops}",
            "--pause-between-programs",
            "--mlir-print-op-generic",
        ],
        text=True,
        stdin=sp.PIPE,
        stdout=sp.PIPE,
    )

    assert enumerator.stdin is not None
    assert enumerator.stdout is not None

    while (func := read_program_from_lines(ctx, enumerator.stdout)) is not None:
        # Send a character to the enumerator to continue.
        enumerator.stdin.write("a")
        enumerator.stdin.flush()

        yield func


def run_module_through_smtlib(module: ModuleOp) -> Any:
    smtlib_program = StringIO()
    print_to_smtlib(module, smtlib_program)

    # Parse the SMT-LIB program and run it through the Z3 solver.
    solver = z3.Solver()
    try:
        solver.from_string(  # pyright: ignore[reportUnknownMemberType]
            smtlib_program.getvalue()
        )
        result = solver.check()  # pyright: ignore[reportUnknownMemberType]
    except z3.z3types.Z3Exception as e:
        print(
            e.value.decode(  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                "UTF-8"
            ),
            end="",
            file=sys.stderr,
        )
        print("The above error happened with the following query:", file=sys.stderr)
        print(smtlib_program.getvalue(), file=sys.stderr)
        raise Exception()
    if result == z3.unknown:
        print("Z3 couldn't solve the following query:", file=sys.stderr)
        print(smtlib_program.getvalue(), file=sys.stderr)
        raise Exception()
    return result


T = TypeVar("T")


def permute(it: Iterable[T], permutation: Sequence[int]) -> list[T]:
    array = list(it)
    assert len(permutation) == len(array)
    assert len(set(permutation)) == len(permutation)
    return [array[i] for i in permutation]


def is_same_behavior_with_z3(
    left: smt.DefineFunOp,
    right: smt.DefineFunOp,
    left_permutation: Sequence[int],
) -> bool:
    if right.func_type.outputs != left.func_type.outputs:
        return False

    module = ModuleOp([])
    builder = Builder(InsertPoint.at_end(module.body.block))

    # Clone both functions into a new module.
    left = builder.insert(left.clone())
    right = builder.insert(right.clone())

    # Declare a variable for each function input.
    left_args: list[SSAValue | None] = [None] * len(left.func_type.inputs)
    right_args: list[SSAValue | None] = [None] * len(right.func_type.inputs)
    # In case one of the programs contain more arguments, the programs have to
    # be equivalent no matter the values of the additional arguments.
    for (can_arg_index, can_arg_ty), (func_arg_index, func_arg_ty) in zip(
        permute(enumerate(left.func_type.inputs), left_permutation),
        enumerate(right.func_type.inputs),
    ):
        if can_arg_ty != func_arg_ty:
            return False
        arg = builder.insert(smt.DeclareConstOp(can_arg_ty)).res
        left_args[can_arg_index] = arg
        right_args[func_arg_index] = arg

    for index, ty in enumerate(left.func_type.inputs):
        if left_args[index] is None:
            left_args[index] = builder.insert(smt.DeclareConstOp(ty)).res

    for index, ty in enumerate(right.func_type.inputs):
        if right_args[index] is None:
            right_args[index] = builder.insert(smt.DeclareConstOp(ty)).res

    left_args_complete = cast(list[SSAValue], left_args)
    right_args_complete = cast(list[SSAValue], right_args)

    # Call each function with the same arguments.
    canonical_call = builder.insert(smt.CallOp(left.ret, left_args_complete)).res
    func_call = builder.insert(smt.CallOp(right.ret, right_args_complete)).res

    # We only support single-result functions for now.
    assert len(canonical_call) == 1
    assert len(func_call) == 1

    # Check if the two results are not equal.
    check = builder.insert(smt.DistinctOp(canonical_call[0], func_call[0])).res
    builder.insert(smt.AssertOp(check))

    # Now that we have the module, run it through the Z3 solver.
    return z3.unsat == run_module_through_smtlib(
        module
    )  # pyright: ignore[reportUnknownVariableType]


def is_same_behavior(left: smt.DefineFunOp, right: smt.DefineFunOp) -> bool:
    for left_permutation in itertools.permutations(range(len(left.func_type.inputs))):
        if is_same_behavior_with_z3(left, right, left_permutation):
            return True
    return False


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

    sp.run(
        [
            "generate-equivalent-boolean-smts",
            f"--max-num-args={args.max_num_args}",
            f"--max-num-ops={args.max_num_ops}",
            f"--bitvector-widths={args.bitvector_widths}",
            f"--out-canonicals={PROGRAM_OUTPUT}",
        ]
    )

    canonicals = list[smt.DefineFunOp]()
    with open(PROGRAM_OUTPUT, "r", encoding="UTF-8") as f:
        while (func := read_program_from_lines(ctx, f)) is not None:
            canonicals.append(func)

    print("\033[1m== Looking for duplicate canonicals ==\033[0m")
    for i, canonical in enumerate(canonicals):
        for other in canonicals[:i]:
            print(
                f"\033[2K {i + 1}/{len(canonicals)} "
                f"({func_to_string(canonical)} vs. {func_to_string(other)})...",
                end="\r",
            )
            if is_same_behavior(canonical, other):
                print(
                    f"\033[2K The following canonical programs are equivalent: "
                    f"{func_to_string(canonical)} and {func_to_string(other)}."
                )
    print(f"Checked all {len(canonicals)} canonical programs.")

    print("\033[1m== Looking for programs with no canonical equivalent ==\033[0m")
    program_count = 0
    for func in enumerate_programs(
        ctx, args.max_num_args, args.max_num_ops, args.bitvector_widths
    ):
        program_count += 1
        print(f"\033[2K {program_count} ({func_to_string(func)})...", end="\r")
        if not any(is_same_behavior(func, canonical) for canonical in canonicals):
            print(
                f"\033[2K The following function has no canonical equivalent: "
                f"{func_to_string(func)}."
            )
    print(f"Checked all {program_count} programs.")


if __name__ == "__main__":
    main()
