#!/usr/bin/env python3

import z3  # pyright: ignore[reportMissingTypeStubs]
import argparse
import itertools
import subprocess as sp
import sys
import time
from dataclasses import dataclass
from functools import cmp_to_key, partial
from io import StringIO
from multiprocessing import Pool
from typing import Any, Callable, Generator, TypeVar

from xdsl.context import Context
from xdsl.ir import Attribute
from xdsl.ir.core import BlockArgument, Operation, OpResult, SSAValue
from xdsl.parser import Parser
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.builder import Builder

import xdsl_smt.dialects.synth_dialect as synth
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_bitvector_dialect as bv
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.dialects.builtin import Builtin, FunctionType, IntAttr, IntegerAttr, ModuleOp
from xdsl.dialects.func import Func, FuncOp, ReturnOp

from xdsl_smt.cli.xdsl_smt_run import build_interpreter, interpret
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
REMOVE_REDUNDANT_PATTERNS = "./mlir-fuzz/build/bin/remove-redundant-patterns"
SMT_MLIR = "./mlir-fuzz/dialects/smt.mlir"
EXCLUDE_SUBPATTERNS_FILE = f"/tmp/exclude-subpatterns-{time.time()}"
USE_CPP = True


T = TypeVar("T")


def list_extract(l: list[T], predicate: Callable[[T], bool]) -> T | None:
    """
    Deletes and returns the first element from the passed list that matches the
    predicate.

    If no element matches the predicate, the list is not modified and `None` is
    returned.
    """
    for i, x in enumerate(l):
        if predicate(x):
            del l[i]
            return x
    return None


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
        case OpResult(op=bv.ConstantOp()):
            return 0
        case OpResult(op=op):
            return 1 + sum(formula_size(operand) for operand in op.operands)
        case x:
            raise ValueError(f"Unknown value: {x}")


def program_size(program: ModuleOp) -> int:
    ret = get_inner_func(program).get_return_op()
    assert ret is not None
    return sum(formula_size(argument) for argument in ret.arguments)


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

    # We do not include the return instruction as part of the pattern. If the
    # program contain instructions with multiple return values, or itself
    # returns multiple values, this may lead to unexpected results.
    operations = list(get_inner_func(program).body.ops)[:-1]
    for i, op in enumerate(operations):
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

    lines.append(f'    rewrite %op{len(operations) - 1} with "rewriter"')
    lines.append("  }")
    lines.append("}")

    lines[body_start_index:body_start_index] = [
        f"    %arg{k} = pdl.operand" for k in used_arguments
    ]
    lines[body_start_index:body_start_index] = [
        f"    %attr.{attr} = pdl.attribute = {attr}" for attr in used_attributes
    ]

    return "\n".join(lines)


def enumerate_programs(
    ctx: Context,
    max_num_args: int,
    num_ops: int,
    bv_widths: str,
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
            f"--smt-bitvector-widths={bv_widths}",
            # Make sure cse is applied
            "--cse",
            # Prevent any non-deterministic behavior (hopefully).
            "--seed=1",
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
        # Send a character to the enumerator to continue.
        assert enumerator.stdin is not None
        enumerator.stdin.write("a")
        enumerator.stdin.flush()

        module = Parser(ctx, program).parse_module(True)

        if program_size(module) != num_ops:
            continue

        # Deduplication.
        attributes = get_inner_func(module).attributes
        if "seed" in attributes:
            del attributes["seed"]
        s = str(module)
        if s in enumerated:
            continue
        enumerated.add(s)

        yield module


def values_of_type(ty: Attribute) -> tuple[list[Attribute], bool]:
    """
    Returns values of the passed type.

    The boolean indicates whether the returned values cover the whole type. If
    `true`, this means all possible values of the type were returned.
    """
    match ty:
        case smt.BoolType():
            return [IntegerAttr.from_bool(False), IntegerAttr.from_bool(True)], True
        case bv.BitVectorType(width=IntAttr(data=width)):
            w = min(2, width)
            return [
                IntegerAttr.from_int_and_width(value, width) for value in range(1 << w)
            ], (w == width)
        case _:
            raise ValueError(f"Unsupported type: {ty}")


def clone_func_to_smt_func(func: FuncOp) -> smt.DefineFunOp:
    """
    Convert a func.func to an smt.define_fun operation.
    Do not mutate the original function.
    """
    new_region = func.body.clone()

    # Replace the func.return with an smt.return operation
    func_return = new_region.block.last_op
    assert isinstance(func_return, ReturnOp)
    rewriter = Rewriter()
    rewriter.insert_op(
        smt.ReturnOp(func_return.arguments), InsertPoint.before(func_return)
    )
    rewriter.erase_op(func_return)

    return smt.DefineFunOp(new_region)


def run_module_through_smtlib(module: ModuleOp) -> Any:
    smtlib_program = StringIO()
    print_to_smtlib(module, smtlib_program)

    # Parse the SMT-LIB program and run it through the z3 solver.
    solver = z3.Solver()
    solver.from_string(  # pyright: ignore[reportUnknownMemberType]
        smtlib_program.getvalue()
    )
    result = solver.check()  # pyright: ignore[reportUnknownMemberType]
    if result == z3.unknown:
        print("z3 couldn't solve the following query:")
        print(smtlib_program.getvalue())
        exit(1)
    return result


def is_same_behavior_with_z3(left: ModuleOp, right: ModuleOp) -> bool:
    """
    Check wether two programs are semantically equivalent using z3.
    We assume that both programs have the same function type.
    """
    func_left = clone_func_to_smt_func(get_inner_func(left))
    func_right = clone_func_to_smt_func(get_inner_func(right))

    function_type = func_left.func_type

    module = ModuleOp([])
    builder = Builder(InsertPoint.at_end(module.body.block))

    # Clone both functions into a new module
    func_left = builder.insert(func_left.clone())
    func_right = builder.insert(func_right.clone())

    # Declare a variable for each function input
    args = list[SSAValue]()
    for arg_type in function_type.inputs:
        arg = builder.insert(smt.DeclareConstOp(arg_type)).res
        args.append(arg)

    # Call each function with the same arguments
    left_call = builder.insert(smt.CallOp(func_left.ret, args)).res
    right_call = builder.insert(smt.CallOp(func_right.ret, args)).res

    # We only support single-result functions for now
    assert len(left_call) == 1

    # Check if the two results are not equal
    check = builder.insert(smt.DistinctOp(left_call[0], right_call[0])).res
    builder.insert(smt.AssertOp(check))

    # Now that we have the module, run it through the z3 solver
    return z3.unsat == run_module_through_smtlib(
        module
    )  # pyright: ignore[reportUnknownVariableType]


@dataclass(eq=True, frozen=True)
class Signature:
    function_type: FunctionType
    results: tuple[Any, ...]
    is_total: bool

    @classmethod
    def from_program(cls, program: ModuleOp):
        """
        Computes a value that highly depends on the behavior of the passed program.
        """
        interpreter = build_interpreter(program, 64)
        function_type = get_inner_func(program).function_type
        values = [values_of_type(ty) for ty in function_type.inputs]
        is_total = all(total for _, total in values)
        arguments = itertools.product(*(vals for vals, _ in values))
        return cls(
            function_type,
            tuple(interpret(interpreter, args) for args in arguments),
            is_total,
        )


def is_same_behavior(left: ModuleOp, right: ModuleOp, signature: Signature) -> bool:
    """
    Tests whether two programs having the same signature are semantically
    equivalent.
    """
    if signature.is_total:
        # The signature covers the whole behavior, so no need to do anything
        # expensive.
        return True

    return is_same_behavior_with_z3(left, right)


def compare_values_lexicographically(left: SSAValue, right: SSAValue) -> int:
    match left, right:
        case BlockArgument(index=i), BlockArgument(index=j):
            return i - j
        case BlockArgument(), OpResult():
            return 1
        case OpResult(), BlockArgument():
            return -1
        case OpResult(op=smt.ConstantBoolOp(value=lv)), OpResult(
            op=smt.ConstantBoolOp(value=rv)
        ):
            return bool(lv) - bool(rv)
        case OpResult(op=lop, index=i), OpResult(op=rop, index=j):
            if isinstance(lop, type(rop)):
                return i - j
            if len(lop.operands) != len(rop.operands):
                return len(lop.operands) - len(rop.operands)
            for lo, ro in zip(lop.operands, rop.operands, strict=True):
                c = compare_values_lexicographically(lo, ro)
                if c != 0:
                    return c
            return 0
        case l, r:
            raise ValueError(f"Unknown value: {l} or {r}")


def compare_programs_lexicographically(left: ModuleOp, right: ModuleOp) -> int:
    if program_size(left) < program_size(right):
        return -1
    if program_size(right) > program_size(left):
        return 1
    left_ret = get_inner_func(left).get_return_op()
    assert left_ret is not None
    right_ret = get_inner_func(right).get_return_op()
    assert right_ret is not None
    return compare_values_lexicographically(
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


Bucket = list[ModuleOp]


@dataclass
class Behavior:
    signature: Signature
    programs: Bucket


@dataclass
class SignedProgram:
    signature: Signature
    program: ModuleOp


def sort_bucket(
    canonicals: list[SignedProgram],
    signed_bucket: tuple[Signature, Bucket],
) -> tuple[list[Behavior], Bucket]:
    signature, bucket = signed_bucket

    # Sort programs into actual behavior buckets.
    behaviors: list[Bucket] = []
    for program in bucket:
        for behavior in behaviors:
            if is_same_behavior(program, behavior[0], signature):
                behavior.append(program)
                break
        else:
            behaviors.append([program])

    # Detect known behaviors.
    illegals: Bucket = []
    for canonical in canonicals:
        if signature != canonical.signature:
            continue
        behavior = list_extract(
            behaviors,
            lambda behavior: is_same_behavior(
                behavior[0], canonical.program, signature
            ),
        )
        if behavior is not None:
            illegals.extend(behavior)

    # The rest are new behaviors.
    new_behaviors = [Behavior(signature, behavior) for behavior in behaviors]

    return new_behaviors, illegals


def sort_programs(
    buckets: dict[Signature, Bucket],
    canonicals: list[SignedProgram],
) -> tuple[list[Behavior], Bucket]:
    """
    Sort programs from the specified buckets into programs with new behaviors,
    and illegal subpatterns.

    The returned pair is `new_behaviors, new_illegals`.
    """

    new_behaviors: list[Behavior] = []
    new_illegals: Bucket = []

    with Pool() as p:
        for i, (behaviors, illegals) in enumerate(
            p.imap_unordered(
                partial(sort_bucket, canonicals),
                buckets.items(),
            )
        ):
            print(f" {round(100.0 * i / len(buckets), 1)} %", end="\r")
            new_behaviors.extend(behaviors)
            new_illegals.extend(illegals)

    return new_behaviors, new_illegals


def pretty_print_value(x: SSAValue, nested: bool):
    infix = isinstance(x, OpResult) and (
        isinstance(x.op, smt.BinaryBoolOp)
        or isinstance(x.op, smt.BinaryTOp)
        or isinstance(x.op, smt.IteOp)
        or isinstance(x.op, bv.BinaryBVOp)
    )
    if infix and nested:
        print("(", end="")
    match x:
        case BlockArgument(index=i, type=smt.BoolType()):
            print(("x", "y", "z", "w", "v", "u", "t", "s")[i], end="")
        case BlockArgument(index=i, type=bv.BitVectorType(width=width)):
            print(("x", "y", "z", "w", "v", "u", "t", "s")[i], end="")
            print(f"[{width.data}]", end="")
        case OpResult(op=smt.ConstantBoolOp(value=val), index=0):
            print("⊤" if val else "⊥", end="")
        case OpResult(op=bv.ConstantOp(value=val), index=0):
            width = val.type.width.data
            value = val.value.data
            print(f"{{:0{width}b}}".format(value), end="")
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
        case OpResult(op=bv.AddOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True)
            print(" + ", end="")
            pretty_print_value(rhs, True)
        case OpResult(op=bv.AndOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True)
            print(" & ", end="")
            pretty_print_value(rhs, True)
        case OpResult(op=bv.OrOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True)
            print(" | ", end="")
            pretty_print_value(rhs, True)
        case OpResult(op=bv.MulOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True)
            print(" * ", end="")
            pretty_print_value(rhs, True)
        case OpResult(op=bv.NotOp(arg=arg), index=0):
            print("~", end="")
            pretty_print_value(arg, True)
        case _:
            raise ValueError(f"Unknown value for pretty print: {x}")
    if infix and nested:
        print(")", end="")


def pretty_print_program(program: ModuleOp):
    ret = get_inner_func(program).get_return_op()
    assert ret is not None
    pretty_print_value(ret.arguments[0], False)
    print()


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

    canonicals: list[SignedProgram] = []
    illegals: list[ModuleOp] = []

    try:
        for m in range(args.max_num_ops + 1):
            print(f"\033[1m== Size {m} ==\033[0m")
            step_start = time.time()

            print("Enumerating programs...")
            new_program_count = 0
            buckets: dict[Signature, list[ModuleOp]] = {}
            for program in enumerate_programs(
                ctx, args.max_num_args, m, args.bitvector_widths, illegals
            ):
                new_program_count += 1
                print(f" {new_program_count}", end="\r")
                signature = Signature.from_program(program)
                if signature not in buckets:
                    buckets[signature] = []
                buckets[signature].append(program)
            print(f"Generated {new_program_count} programs of this size.")

            print("Sorting programs...")
            new_behaviors, new_illegals = sort_programs(buckets, canonicals)
            print(
                f"Found {len(new_behaviors)} new behaviors, "
                f"exhibited by {sum(len(behavior.programs) for behavior in new_behaviors)} programs."
            )

            print("Choosing new canonical programs...")
            for behavior in new_behaviors:
                behavior.programs.sort(
                    key=cmp_to_key(compare_programs_lexicographically)
                )
                canonical = behavior.programs[0]
                canonicals.append(SignedProgram(behavior.signature, canonical))
                new_illegals.extend(
                    program
                    for program in behavior.programs
                    if not is_pattern(program, canonical)
                )
            print(f"Found {len(new_illegals)} new illegal subpatterns.")

            print("Removing redundant subpatterns...")
            if USE_CPP:
                input = StringIO()
                print("module {", file=input)
                for illegal in new_illegals:
                    print(illegal, file=input)
                print("}", file=input)
                cpp_res = sp.run(
                    [REMOVE_REDUNDANT_PATTERNS],
                    input=input.getvalue(),
                    stdout=sp.PIPE,
                    text=True,
                )
                removed_indices: list[int] = []
                for idx, line in enumerate(cpp_res.stdout.splitlines()):
                    if line == "true":
                        removed_indices.append(idx)
                redundant_count = len(removed_indices)
                for idx in reversed(removed_indices):
                    del new_illegals[idx]
            else:
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
            print(
                f"We now have a total of {len(canonicals)} behaviors and {len(illegals)} illegal subpatterns."
            )

        # Write results to disk.
        old_stdout = sys.stdout
        with open(args.out_file, "w", encoding="UTF-8") as f:
            sys.stdout = f
            for program in canonicals:
                pretty_print_program(program.program)
            print("// -----")
            for program in illegals:
                pretty_print_program(program)
        sys.stdout = old_stdout

        if args.summary:
            print(f"\033[1m== Summary (canonical programs) ==\033[0m")
            for program in canonicals:
                pretty_print_program(program.program)

    except BrokenPipeError as e:
        # The enumerator has terminated
        pass
    except Exception as e:
        print(f"Error while enumerating programs: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)


if __name__ == "__main__":
    main()
