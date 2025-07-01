#!/usr/bin/env python3

import argparse
import itertools
import subprocess as sp
import sys
import time
import z3  # pyright: ignore[reportMissingTypeStubs]
from dataclasses import dataclass
from functools import partial
from io import StringIO
from multiprocessing import Pool
from typing import Any, Callable, Generator, Generic, Iterable, Sequence, TypeVar, cast

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
from xdsl.dialects.builtin import Builtin, IntAttr, IntegerAttr, ModuleOp
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


Result = tuple[Any, ...]


class FrozenMultiset(Generic[T]):
    _contents: frozenset[tuple[T, int]]

    def __init__(self, values: Iterable[T]):
        items = dict[T, int]()
        for value in values:
            if value in items:
                items[value] += 1
            else:
                items[value] = 1
        self._contents = frozenset(items.items())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FrozenMultiset):
            return False
        return self._contents.__eq__(cast(FrozenMultiset[Any], other)._contents)

    def __hash__(self) -> int:
        return self._contents.__hash__()


@dataclass(frozen=True, slots=True)
class Signature:
    """
    A value that can be computed from a program, and highly depends on its
    behavior.

    The signatures of two semantically equivalent programs are guaranteed to
    compare equal. Furthermore, if two programs are semantically equivalent
    after removing inputs that don't affect the output, their signatures are
    guaranteed to compare equal as well.
    """

    _useful_inputs: FrozenMultiset[Attribute]
    _outputs: tuple[Attribute, ...]
    _results_with_permutations: FrozenMultiset[tuple[Result, ...]]


Permutation = Sequence[int]


def permute(seq: Sequence[T], permutation: Permutation) -> tuple[T, ...]:
    assert len(seq) == len(permutation)
    return tuple(seq[i] for i in permutation)


def reverse_permute(seq: Sequence[T], permutation: Permutation) -> tuple[T, ...]:
    assert len(seq) == len(permutation)
    result: list[T | None] = [None for _ in seq]
    for x, i in zip(seq, permutation, strict=True):
        result[i] = x
    assert all(x is not None for x in result)
    return tuple(cast(list[T], result))


class Program:
    module: ModuleOp
    _size: int
    _input_cardinalities: tuple[int, ...]
    _base_results: tuple[Result, ...]
    _signature: Signature
    _is_signature_total: bool
    _useless_input_mask: tuple[bool, ...]

    @staticmethod
    def _formula_size(formula: SSAValue) -> int:
        match formula:
            case BlockArgument():
                return 0
            case OpResult(op=smt.ConstantBoolOp()):
                return 0
            case OpResult(op=bv.ConstantOp()):
                return 0
            case OpResult(op=op):
                return 1 + sum(
                    Program._formula_size(operand) for operand in op.operands
                )
            case x:
                raise ValueError(f"Unknown value: {x}")

    @staticmethod
    def _values_of_type(ty: Attribute) -> tuple[list[Attribute], bool]:
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
                    IntegerAttr.from_int_and_width(value, width)
                    for value in range(1 << w)
                ], (w == width)
            case _:
                raise ValueError(f"Unsupported type: {ty}")

    def _input_permutations(self) -> Iterable[Permutation]:
        assert self._useless_input_mask is not None
        useless_indices = set(i for i, m in enumerate(self._useless_input_mask) if m)
        for permutation in itertools.permutations(range(self.arity())):
            if all(permutation[i] == i for i in useless_indices):
                yield permutation

    def _results_with_permutation(self, permutation: Permutation) -> tuple[Result, ...]:
        assert self._input_cardinalities is not None
        assert self._base_results is not None
        # This could be achieved using less memory with some arithmetic.
        input_ids = itertools.product(*(range(c) for c in self._input_cardinalities))
        indices: dict[tuple[int, ...], int] = {
            iid: result_index for result_index, iid in enumerate(input_ids)
        }
        permuted_input_ids = itertools.product(
            *permute([range(c) for c in self._input_cardinalities], permutation)
        )
        return tuple(
            self._base_results[indices[reverse_permute(piid, permutation)]]
            for piid in permuted_input_ids
        )

    def __init__(self, module: ModuleOp):
        self.module = module

        self._size = sum(
            Program._formula_size(argument) for argument in self.ret().arguments
        )

        arity = self.arity()
        interpreter = build_interpreter(self.module, 64)
        function_type = self.func().function_type
        values_for_each_input = [
            Program._values_of_type(ty) for ty in function_type.inputs
        ]
        self._is_signature_total = all(total for _, total in values_for_each_input)

        # First, detect inputs that don't affect the results within the set of
        # inputs that we check.
        results_for_fixed_inputs: list[dict[tuple[Attribute, ...], set[Result]]] = [
            {
                other_inputs: set()
                for other_inputs in itertools.product(
                    *(
                        vals
                        for vals, _ in values_for_each_input[:i]
                        + values_for_each_input[i + 1 :]
                    )
                )
            }
            for i in range(arity)
        ]
        for inputs in itertools.product(*(vals for vals, _ in values_for_each_input)):
            result = interpret(interpreter, inputs)
            for i in range(arity):
                results_for_fixed_inputs[i][inputs[:i] + inputs[i + 1 :]].add(result)
        assert all(
            all(len(results) != 0 for results in results_for_fixed_input.values())
            for results_for_fixed_input in results_for_fixed_inputs
        )
        input_useless_here = [
            all(
                len(results) == 1 and len(values_for_each_input[i]) != 1
                for results in results_for_fixed_input.values()
            )
            for i, results_for_fixed_input in enumerate(results_for_fixed_inputs)
        ]

        # Then, compute which of those inputs are actually useless.
        self._useless_input_mask = tuple(
            input_useless_here[i]
            and (self._is_signature_total or is_input_useless_z3(self, i))
            for i in range(arity)
        )
        useful_inputs = FrozenMultiset(
            ty for ty, m in zip(function_type.inputs, self._useless_input_mask) if not m
        )

        # Now, compute the results ignoring useless inputs.
        values_for_each_useful_input = [
            (
                [values_for_each_input[i][0][0]]
                if self._useless_input_mask[i]
                else values_for_each_input[i][0]
            )
            for i in range(arity)
        ]
        self._base_results = tuple(
            interpret(interpreter, inputs)
            for inputs in itertools.product(
                *(vals for vals in values_for_each_useful_input)
            )
        )

        # Finally, compute the outputs for all permutations of useful inputs.
        self._input_cardinalities = tuple(
            len(values) for values in values_for_each_useful_input
        )
        results_with_permutations = FrozenMultiset(
            self._results_with_permutation(permutation)
            for permutation in self._input_permutations()
        )

        self._signature = Signature(
            useful_inputs,
            tuple(function_type.outputs),
            results_with_permutations,
        )

    def func(self) -> FuncOp:
        """Returns the underlying function."""
        assert len(self.module.ops) == 1
        assert isinstance(self.module.ops.first, FuncOp)
        return self.module.ops.first

    def ret(self) -> ReturnOp:
        """Returns the return operation of the underlying function."""
        r = self.func().get_return_op()
        assert r is not None
        return r

    def arity(self) -> int:
        return len(self.func().function_type.inputs)

    def size(self) -> int:
        return self._size

    def signature(self) -> Signature:
        return self._signature

    def is_signature_total(self) -> bool:
        """
        Whether the whole behavior of this program is encapsulated in its
        signature. If two programs have a total signature, they are equivalent
        if, and only if, their signatures compare equal.
        """
        return self._is_signature_total

    def useless_input_mask(self) -> tuple[bool, ...]:
        """
        Booleans indicating, for each corresponding function input, whether the
        input is useless. A useless input is an input whose value does not
        affect the outputs.
        """
        return self._useless_input_mask

    def permuted_useful_inputs(
        self, permutation: Permutation
    ) -> list[tuple[int, Attribute]]:
        """
        Returns the indices and types of the non-useless inputs in order after
        applying the specified permutation to all inputs.
        """
        return [
            (i, ty)
            for i, (ty, useless) in permute(
                list(
                    enumerate(
                        zip(
                            self.func().function_type.inputs,
                            self.useless_input_mask(),
                            strict=True,
                        )
                    )
                ),
                permutation,
            )
            if not useless
        ]

    def useful_inputs(self) -> list[tuple[int, Attribute]]:
        """
        Returns the indices and types of the non-useless inputs in order.
        """
        return self.permuted_useful_inputs(range(self.arity()))

    @staticmethod
    def _compare_values_lexicographically(left: SSAValue, right: SSAValue) -> int:
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
                    c = Program._compare_values_lexicographically(lo, ro)
                    if c != 0:
                        return c
                return 0
            case l, r:
                raise ValueError(f"Unknown value: {l} or {r}")

    def _compare_lexicographically(self, other: "Program") -> int:
        if self.size() < other.size():
            return -1
        if self.size() > other.size():
            return 1
        self_outs = self.ret().arguments
        other_outs = self.ret().arguments
        if len(self_outs) < len(other_outs):
            return -1
        if len(self_outs) > len(other_outs):
            return 1
        for self_out, other_out in zip(self_outs, other_outs, strict=True):
            c = Program._compare_values_lexicographically(self_out, other_out)
            if c != 0:
                return c
        return 0

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Program):
            return NotImplemented
        return self._compare_lexicographically(other) < 0

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, Program):
            return NotImplemented
        return self._compare_lexicographically(other) <= 0

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Program):
            return NotImplemented
        return self._compare_lexicographically(other) > 0

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, Program):
            return NotImplemented
        return self._compare_lexicographically(other) >= 0

    def is_same_behavior(self, other: "Program") -> bool:
        """
        Tests whether two programs are semantically equivalent ignoring useless
        arguments.
        """

        if self.signature() != other.signature():
            return False

        if self.is_signature_total() and other.is_signature_total():
            return True

        for permutation in self._input_permutations():
            # First test whether this permutation has a chance to work.
            if self._results_with_permutation(permutation) == other._base_results:
                # Only then, resort to Z3.
                if is_same_behavior_with_z3(self, other, permutation):
                    return True

        return False

    @staticmethod
    def _unify_value(
        pattern_value: SSAValue,
        program_value: SSAValue,
        pattern_argument_values: list[SSAValue | None] | None,
    ) -> bool:
        match pattern_value, program_value:
            case BlockArgument(index=i), x:
                # If `left_argument_values` is None, then we should match the
                # LHS as is.
                if pattern_argument_values is None:
                    return isinstance(x, BlockArgument) and x.index == i
                # Otherwise, we "unify" the LHS argument with the corresponding
                # program value.
                expected_value = pattern_argument_values[i]
                if expected_value is None:
                    pattern_argument_values[i] = x
                    return True
                return Program._unify_value(expected_value, x, None)
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
                    if not Program._unify_value(
                        lhs_operand, prog_operand, pattern_argument_values
                    ):
                        return False
                return True
            case _:
                return False

    def is_pattern(self, other: "Program") -> bool:
        """
        Tests whether this program is a pattern (LHS) of the other program.
        """
        if len(self.ret().arguments) != len(other.ret().arguments):
            return False
        argument_values: list[SSAValue | None] = [None] * len(self.func().args)
        return all(
            Program._unify_value(s, o, argument_values)
            for s, o in zip(
                self.ret().arguments,
                other.ret().arguments,
                strict=True,
            )
        )

    def to_pdl_pattern(self) -> str:
        """Creates a PDL pattern from this program."""

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

        # We do not include the return instruction as part of the pattern. If
        # the program contains instructions with multiple return values, or
        # itself returns multiple values, this may lead to unexpected results.
        operations = list(self.func().body.ops)[:-1]
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

        # To end the pattern with a line ending.
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _pretty_print_value(x: SSAValue, nested: bool):
        infix = isinstance(x, OpResult) and len(x.op.operand_types) > 1
        parenthesized = infix and nested
        if parenthesized:
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
                Program._pretty_print_value(arg, True)
            case OpResult(op=smt.AndOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, True)
                print(" ∧ ", end="")
                Program._pretty_print_value(rhs, True)
            case OpResult(op=smt.OrOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, True)
                print(" ∨ ", end="")
                Program._pretty_print_value(rhs, True)
            case OpResult(op=smt.ImpliesOp(lhs=lhs, rhs=rhs), index=0):
                Program._pretty_print_value(lhs, True)
                print(" → ", end="")
                Program._pretty_print_value(rhs, True)
            case OpResult(op=smt.DistinctOp(lhs=lhs, rhs=rhs), index=0):
                Program._pretty_print_value(lhs, True)
                print(" ≠ ", end="")
                Program._pretty_print_value(rhs, True)
            case OpResult(op=smt.EqOp(lhs=lhs, rhs=rhs), index=0):
                Program._pretty_print_value(lhs, True)
                print(" = ", end="")
                Program._pretty_print_value(rhs, True)
            case OpResult(op=smt.XOrOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, True)
                print(" ⊕ ", end="")
                Program._pretty_print_value(rhs, True)
            case OpResult(
                op=smt.IteOp(cond=cond, true_val=true_val, false_val=false_val), index=0
            ):
                Program._pretty_print_value(cond, True)
                print(" ? ", end="")
                Program._pretty_print_value(true_val, True)
                print(" : ", end="")
                Program._pretty_print_value(false_val, True)
            case OpResult(op=bv.AddOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, True)
                print(" + ", end="")
                Program._pretty_print_value(rhs, True)
            case OpResult(op=bv.AndOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, True)
                print(" & ", end="")
                Program._pretty_print_value(rhs, True)
            case OpResult(op=bv.OrOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, True)
                print(" | ", end="")
                Program._pretty_print_value(rhs, True)
            case OpResult(op=bv.MulOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, True)
                print(" * ", end="")
                Program._pretty_print_value(rhs, True)
            case OpResult(op=bv.NotOp(arg=arg), index=0):
                print("~", end="")
                Program._pretty_print_value(arg, True)
            case _:
                raise ValueError(f"Unknown value for pretty print: {x}")
        if parenthesized:
            print(")", end="")

    def pretty_print(self):
        Program._pretty_print_value(self.ret().arguments[0], False)
        print()


Bucket = list[Program]


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
    ctx: Context,
    max_num_args: int,
    num_ops: int,
    bv_widths: str,
    illegals: list[Program],
) -> Generator[Program, None, None]:
    with open(EXCLUDE_SUBPATTERNS_FILE, "w") as f:
        for program in illegals:
            f.write(program.to_pdl_pattern())
            f.write("// -----\n")

    enumerator = sp.Popen(
        [
            MLIR_ENUMERATE,
            SMT_MLIR,
            "--configuration=smt",
            f"--smt-bitvector-widths={bv_widths}",
            # Make sure CSE is applied.
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

    while (source := read_program_from_enumerator(enumerator)) is not None:
        # Send a character to the enumerator to continue.
        assert enumerator.stdin is not None
        enumerator.stdin.write("a")
        enumerator.stdin.flush()

        program = Program(Parser(ctx, source).parse_module(True))

        if program.size() != num_ops:
            continue

        yield program


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


def is_same_behavior_with_z3(
    left: Program,
    right: Program,
    left_permutation: Permutation,
) -> bool:
    """
    Check wether two programs are semantically equivalent after permuting the
    arguments of the left program, using Z3.
    """

    func_left = clone_func_to_smt_func(left.func())
    func_right = clone_func_to_smt_func(right.func())

    module = ModuleOp([])
    builder = Builder(InsertPoint.at_end(module.body.block))

    # Clone both functions into a new module.
    func_left = builder.insert(func_left.clone())
    func_right = builder.insert(func_right.clone())

    # Declare a variable for each function input.
    args_left: list[SSAValue | None] = [None] * len(func_left.func_type.inputs)
    args_right: list[SSAValue | None] = [None] * len(func_right.func_type.inputs)
    for (left_index, left_type), (right_index, right_type) in zip(
        left.permuted_useful_inputs(left_permutation), right.useful_inputs()
    ):
        assert left_type == right_type, "Function inputs do not match."
        arg = builder.insert(smt.DeclareConstOp(left_type)).res
        args_left[left_index] = arg
        args_right[right_index] = arg

    for index, type in enumerate(func_left.func_type.inputs):
        if args_left[index] is None:
            args_left[index] = builder.insert(smt.DeclareConstOp(type)).res

    for index, type in enumerate(func_right.func_type.inputs):
        if args_right[index] is None:
            args_right[index] = builder.insert(smt.DeclareConstOp(type)).res

    args_left_complete = cast(list[SSAValue], args_left)
    args_right_complete = cast(list[SSAValue], args_right)

    # Call each function with the same arguments.
    left_call = builder.insert(smt.CallOp(func_left.ret, args_left_complete)).res
    right_call = builder.insert(smt.CallOp(func_right.ret, args_right_complete)).res

    # We only support single-result functions for now.
    assert len(left_call) == 1

    # Check if the two results are not equal.
    check = builder.insert(smt.DistinctOp(left_call[0], right_call[0])).res
    builder.insert(smt.AssertOp(check))

    # Now that we have the module, run it through the Z3 solver.
    return z3.unsat == run_module_through_smtlib(
        module
    )  # pyright: ignore[reportUnknownVariableType]


def is_input_useless_z3(program: Program, arg_index: int) -> bool:
    """
    Use Z3 to check wether if the argument at `arg_index` is irrelevant in
    the computation of the function.
    """

    func = program.func()

    module = ModuleOp([])
    builder = Builder(InsertPoint.at_end(module.body.block))

    # Clone the function twice into the new module
    func1 = builder.insert(clone_func_to_smt_func(func))
    func2 = builder.insert(func1.clone())
    function_type = func1.func_type

    # Declare one variable for each function input, and two for the input we
    # want to check.
    args1 = list[SSAValue]()
    args2 = list[SSAValue]()

    for i, arg_type in enumerate(function_type.inputs):
        arg1 = builder.insert(smt.DeclareConstOp(arg_type)).res
        args1.append(arg1)
        if i == arg_index:
            # We declare two variables for the argument we want to check.
            arg2 = builder.insert(smt.DeclareConstOp(arg_type)).res
            args2.append(arg2)
        else:
            args2.append(arg1)

    # Call the functions with their set of arguments
    call1 = builder.insert(smt.CallOp(func1.ret, args1)).res
    call2 = builder.insert(smt.CallOp(func2.ret, args2)).res

    assert len(call1) == 1, "Only single-result functions are supported."

    # Check if the two results are not equal
    check = builder.insert(smt.DistinctOp(call1[0], call2[0])).res
    builder.insert(smt.AssertOp(check))

    return z3.unsat == run_module_through_smtlib(
        module
    )  # pyright: ignore[reportUnknownVariableType]


def sort_bucket(
    canonicals: list[Program],
    bucket: Bucket,
) -> tuple[list[Bucket], list[Bucket]]:
    # All programs in the bucket have the same signature.
    signature = bucket[0].signature()

    # Sort programs into actual behavior buckets.
    behaviors: list[Bucket] = []
    for program in bucket:
        for behavior in behaviors:
            if program.is_same_behavior(behavior[0]):
                behavior.append(program)
                break
        else:
            behaviors.append([program])

    # Detect known behaviors. The rest are new behaviors.
    known_behaviors: list[Bucket] = []
    for canonical in canonicals:
        if signature != canonical.signature():
            continue
        behavior = list_extract(
            behaviors,
            lambda behavior: behavior[0].is_same_behavior(canonical),
        )
        if behavior is not None:
            known_behaviors.append(behavior)

    return behaviors, known_behaviors


def sort_programs(
    buckets: list[Bucket],
    canonicals: list[Program],
) -> tuple[list[Bucket], list[Bucket]]:
    """
    Sort programs from the specified buckets into programs with new behaviors,
    and illegal subpatterns.

    The returned pair is `new_behaviors, known_behaviors`.
    """

    new_behaviors: list[Bucket] = []
    known_behaviors: list[Bucket] = []

    with Pool() as p:
        for i, (new, known) in enumerate(
            p.imap_unordered(partial(sort_bucket, canonicals), buckets)
        ):
            print(f"\033[2K {round(100.0 * i / len(buckets), 1)} %", end="\r")
            new_behaviors.extend(new)
            known_behaviors.extend(known)

    return new_behaviors, known_behaviors


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

    canonicals: list[Program] = []
    illegals: list[Program] = []

    try:
        for m in range(args.max_num_ops + 1):
            print(f"\033[1m== Size {m} ==\033[0m")
            step_start = time.time()

            print("Enumerating programs...")
            new_program_count = 0
            buckets: dict[Signature, Bucket] = {}
            for program in enumerate_programs(
                ctx, args.max_num_args, m, args.bitvector_widths, illegals
            ):
                new_program_count += 1
                print(f"\033[2K {new_program_count}", end="\r")
                signature = program.signature()
                if signature not in buckets:
                    buckets[signature] = []
                buckets[signature].append(program)
            print(f"\033[2KGenerated {new_program_count} programs of this size.")

            print("Sorting programs...")
            new_behaviors, known_behaviors = sort_programs(
                list(buckets.values()), canonicals
            )
            new_illegals = [
                program for behavior in known_behaviors for program in behavior
            ]
            print(
                f"\033[2KFound {len(new_behaviors)} new behaviors, "
                f"exhibited by {sum(len(behavior) for behavior in new_behaviors)} programs."
            )

            print("Choosing new canonical programs...")
            new_canonicals = []
            for behavior in new_behaviors:
                behavior.sort()
                canonical = behavior[0]
                new_canonicals.append(canonical)
            new_illegals.extend(
                program
                for program in behavior
                for behavior in new_behaviors
                if not any(
                    program.is_pattern(canonical) for canonical in new_canonicals
                )
            )
            canonicals.extend(new_canonicals)
            print(f"Found {len(new_illegals)} new illegal subpatterns.")

            print("Removing redundant subpatterns...")
            input = StringIO()
            print("module {", file=input)
            for illegal in new_illegals:
                print(illegal.module, file=input)
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
            illegals.extend(new_illegals)
            print(f"Removed {redundant_count} redundant illegal subpatterns.")

            step_end = time.time()
            print(f"Finished step in {round(step_end - step_start, 2):.02f} s.")
            print(
                f"We now have a total of {len(canonicals)} behaviors and {len(illegals)} illegal subpatterns."
            )

        # Write results to disk.
        old_stdout = sys.stdout
        with open(args.out_file, "w", encoding="UTF-8") as f:
            sys.stdout = f
            for program in canonicals:
                program.pretty_print()
            print("// -----")
            for program in illegals:
                program.pretty_print()
        sys.stdout = old_stdout

        if args.summary:
            print(f"\033[1m== Summary (canonical programs) ==\033[0m")
            for program in canonicals:
                program.pretty_print()

    except BrokenPipeError:
        # The enumerator has terminated
        pass
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)


if __name__ == "__main__":
    main()
