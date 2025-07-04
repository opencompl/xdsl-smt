#!/usr/bin/env python3

import argparse
import copy
import itertools
import subprocess as sp
import sys
import time
import z3  # pyright: ignore[reportMissingTypeStubs]
from dataclasses import dataclass
from functools import partial
from io import StringIO
from multiprocessing import Pool
from typing import Any, Generator, Generic, IO, Iterable, Sequence, TypeVar, cast

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
        "--out-canonicals",
        type=str,
        help="the file in which to write the generated canonical programs",
        default="",
    )
    arg_parser.add_argument(
        "--out-rewrites",
        type=str,
        help="the file in which to write the generated rewrite rules",
        default="",
    )
    arg_parser.add_argument(
        "--summarize-canonicals",
        dest="summarize_canonicals",
        action="store_true",
        help="if present, prints a human-readable summary of the generated canonical programs",
    )

    arg_parser.add_argument(
        "--summarize-rewrites",
        dest="summarize_rewrites",
        action="store_true",
        help="if present, prints a human-readable summary of the generated rewrite rules",
    )


MLIR_ENUMERATE = "./mlir-fuzz/build/bin/mlir-enumerate"
REMOVE_REDUNDANT_PATTERNS = "./mlir-fuzz/build/bin/remove-redundant-patterns"
SMT_MLIR = "./mlir-fuzz/dialects/smt.mlir"
EXCLUDE_SUBPATTERNS_FILE = f"/tmp/exclude-subpatterns-{time.time()}"
BUILDING_BLOCKS_FILE = f"/tmp/building-blocks-{time.time()}"


T = TypeVar("T")


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

    def __repr__(self) -> str:
        items: list[T] = []
        for item, count in self._contents:
            items.extend([item] * count)
        return f"FrozenMultiset({items!r})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FrozenMultiset):
            return False
        return self._contents.__eq__(cast(FrozenMultiset[Any], other)._contents)

    def __hash__(self) -> int:
        return self._contents.__hash__()


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """
    A value that can be computed from a program, and highly depends on its
    behavior.

    The fingerprints of two semantically equivalent programs are guaranteed to
    compare equal. Furthermore, if two programs are semantically equivalent
    after removing inputs that don't affect the output, their fingerprints are
    guaranteed to compare equal as well.
    """

    _useful_input_types: FrozenMultiset[Attribute]
    _output_types: tuple[Attribute, ...]
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
    _module: ModuleOp
    _size: int
    _input_cardinalities: tuple[int, ...]
    """
    The cardinality of each input, before applying the permutation.
    """
    _base_results: tuple[Result, ...]
    """
    The results, as computed in order before applying the permutation.
    """
    _fingerprint: Fingerprint
    _is_basic: bool
    _param_permutation: Permutation
    _useless_param_count: int
    """
    The number of useless parameters. Useless parameters are always the last
    ones after apply the permutation.
    """

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

    def permute_useful_parameters(self, permutation: Permutation) -> "Program":
        arity = self.useful_arity()
        assert len(permutation) == arity
        permuted = copy.copy(self)
        permuted._param_permutation = list(permuted._param_permutation)
        permuted._param_permutation[:arity] = [
            permuted._param_permutation[i] for i in permutation
        ]
        return permuted

    def _parameter_permutations(self) -> Iterable["Program"]:
        for permutation in itertools.permutations(range(self.useful_arity())):
            yield self.permute_useful_parameters(permutation)

    def _results(self) -> tuple[Result, ...]:
        assert self._input_cardinalities is not None
        assert self._base_results is not None
        # This could be achieved using less memory with some arithmetic.
        input_ids = itertools.product(*(range(c) for c in self._input_cardinalities))
        indices: dict[tuple[int, ...], int] = {
            iid: result_index for result_index, iid in enumerate(input_ids)
        }
        permuted_input_ids = itertools.product(
            *permute(
                [range(c) for c in self._input_cardinalities], self._param_permutation
            )
        )
        return tuple(
            self._base_results[indices[reverse_permute(piid, self._param_permutation)]]
            for piid in permuted_input_ids
        )

    def __init__(self, module: ModuleOp):
        self._module = module

        self._size = sum(
            Program._formula_size(argument) for argument in self.ret().arguments
        )

        arity = self.arity()
        interpreter = build_interpreter(self._module, 64)
        function_type = self.func().function_type
        values_for_each_param = [
            Program._values_of_type(ty) for ty in function_type.inputs
        ]
        self._is_basic = all(total for _, total in values_for_each_param)

        # First, detect inputs that don't affect the results within the set of
        # inputs that we check.
        results_for_fixed_inputs: list[dict[tuple[Attribute, ...], set[Result]]] = [
            {
                other_inputs: set()
                for other_inputs in itertools.product(
                    *(
                        vals
                        for vals, _ in values_for_each_param[:i]
                        + values_for_each_param[i + 1 :]
                    )
                )
            }
            for i in range(arity)
        ]
        for inputs in itertools.product(*(vals for vals, _ in values_for_each_param)):
            result = interpret(interpreter, inputs)
            for i in range(arity):
                results_for_fixed_inputs[i][inputs[:i] + inputs[i + 1 :]].add(result)
        assert all(
            all(len(results) != 0 for results in results_for_fixed_input.values())
            for results_for_fixed_input in results_for_fixed_inputs
        )
        param_useless_here = [
            all(
                len(results) == 1 and len(values_for_each_param[i]) != 1
                for results in results_for_fixed_input.values()
            )
            for i, results_for_fixed_input in enumerate(results_for_fixed_inputs)
        ]

        # Then, compute which of those parameters are actually useless.
        useless_param_mask = tuple(
            param_useless_here[i]
            and (self._is_basic or is_parameter_useless_z3(self, i))
            for i in range(arity)
        )
        useful_input_types = FrozenMultiset(
            ty for ty, m in zip(function_type.inputs, useless_param_mask) if not m
        )
        self._useless_param_count = sum(useless_param_mask)
        self._param_permutation = [
            i for i, m in enumerate(useless_param_mask) if not m
        ] + [i for i, m in enumerate(useless_param_mask) if m]

        # Now, compute the results ignoring useless parameters.
        values_for_each_useful_param = [
            (
                [values_for_each_param[i][0][0]]
                if useless_param_mask[i]
                else values_for_each_param[i][0]
            )
            for i in range(arity)
        ]
        self._base_results = tuple(
            interpret(interpreter, inputs)
            for inputs in itertools.product(
                *(vals for vals in values_for_each_useful_param)
            )
        )

        # Finally, compute the outputs for all permutations of useful inputs.
        self._input_cardinalities = tuple(
            len(values) for values in values_for_each_useful_param
        )
        results_with_permutations = FrozenMultiset(
            permuted._results() for permuted in self._parameter_permutations()
        )

        self._fingerprint = Fingerprint(
            useful_input_types,
            tuple(function_type.outputs),
            results_with_permutations,
        )

    def module(self) -> ModuleOp:
        """
        Returns the underlying module. Keep in mind that the arguments are not
        permuted in the inner function.
        """
        return self._module

    def func(self) -> FuncOp:
        """
        Returns the underlying function. Keep in mind that the arguments are not
        permuted in the function.
        """
        assert len(self._module.ops) == 1
        assert isinstance(self._module.ops.first, FuncOp)
        return self._module.ops.first

    def ret(self) -> ReturnOp:
        """Returns the return operation of the underlying function."""
        r = self.func().get_return_op()
        assert r is not None
        return r

    def arity(self) -> int:
        return len(self.func().function_type.inputs)

    def useful_arity(self) -> int:
        """Returns the number of useful parameters."""
        return self.arity() - self._useless_param_count

    def size(self) -> int:
        return self._size

    def fingerprint(self) -> Fingerprint:
        return self._fingerprint

    def is_basic(self) -> bool:
        """
        Whether the whole behavior of this program is encapsulated in its
        fingerprint. If two programs have are basic, they are equivalent if, and
        only if, their fingerprints compare equal.
        """
        return self._is_basic

    def useful_parameters(self) -> list[tuple[int, Attribute]]:
        """
        Returns the indices and types of the non-useless parameters in order.
        """

        return [
            (i, ty)
            for i, ty in permute(
                list(enumerate(self.func().function_type.inputs)),
                self._param_permutation,
            )
        ][: self.useful_arity()]

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
        # TODO: Maybe this should take permutation into account?
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
        Tests whether two programs are logically equivalent up to parameter
        permutation, and ignoring useless parameters.
        """

        if self.fingerprint() != other.fingerprint():
            return False

        if self.is_basic() and other.is_basic():
            return True

        for permuted in self._parameter_permutations():
            # First test whether this permutation has a chance to work.
            if permuted._results() == other._results():
                # Only then, resort to Z3.
                if is_same_behavior_with_z3(permuted, other):
                    return True

        return False

    def permute_parameters_to_match(self, other: "Program") -> "Program  | None":
        """
        Returns an identical program with permuted parameters that is logically
        equivalent to the other program, ignoring useless arguments. If no such
        program exists, returns `None`.
        """

        if self.fingerprint() != other.fingerprint():
            return None

        for permuted in self._parameter_permutations():
            if permuted._results() == other._results():
                if (
                    self.is_basic()
                    and other.is_basic()
                    or is_same_behavior_with_z3(permuted, other)
                ):
                    return permuted

        return None

    def to_pdl_pattern(self) -> str:
        """Creates a PDL pattern from this program."""

        lines = [
            "builtin.module {",
            "  pdl.pattern : benefit(1) {",
        ]

        body_start_index = len(lines)

        used_arguments: set[int] = set()
        used_attributes: dict[Attribute, int] = {}
        used_types: dict[Attribute, int] = {}
        op_ids: dict[Operation, int] = {}

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
            properties: list[str] = []
            for name, attribute in op.properties.items():
                if attribute not in used_attributes:
                    used_attributes[attribute] = len(used_attributes)
                properties.append(f'"{name}" = %attr{used_attributes[attribute]}')
            props = "" if len(properties) == 0 else f" {{{', '.join(properties)}}}"
            result_type_ids: list[int] = []
            for ty in op.result_types:
                if ty not in used_types:
                    used_types[ty] = len(used_types)
                result_type_ids.append(used_types[ty])
            outs = (
                f" -> ({', '.join(f'%type{tid}' for tid in result_type_ids)} : {', '.join('!pdl.type' for _ in op.results)})"
                if len(op.results) != 0
                else ""
            )
            lines.append(f'    %op{i} = pdl.operation "{op.name}"{ins}{props}{outs}')
            for j in range(len(op.results)):
                lines.append(f"    %res{i}.{j} = pdl.result {j} of %op{i}")

        ret = self.ret()
        assert len(ret.operands) == 1
        ret_val = ret.operands[0]
        # TODO: In case `ret_val` is a `BlockArgument`, create a pattern that
        # matches anything.
        assert isinstance(
            ret_val, OpResult
        ), "Unable to generate pattern for program with non-op return value"
        lines.append(f'    rewrite %op{op_ids[ret_val.op]} with "rewriter"')

        lines.append("  }")
        lines.append("}")

        argument_type_ids: list[int] = []
        for k in used_arguments:
            ty = self.func().args[k].type
            if ty not in used_types:
                used_types[ty] = len(used_types)
            argument_type_ids.append(used_types[ty])

        lines[body_start_index:body_start_index] = [
            f"    %arg{k} = pdl.operand : %type{tid}"
            for k, tid in zip(used_arguments, argument_type_ids, strict=True)
        ]
        lines[body_start_index:body_start_index] = [
            f"    %type{type_id} = pdl.type : {ty}"
            for ty, type_id in used_types.items()
        ]
        lines[body_start_index:body_start_index] = [
            f"    %attr{attr_id} = pdl.attribute = {attr}"
            for attr, attr_id in used_attributes.items()
        ]

        # To end the pattern with a line ending.
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _pretty_print_value(
        x: SSAValue, permutation: Permutation, nested: bool, *, file: IO[str]
    ):
        infix = isinstance(x, OpResult) and len(x.op.operand_types) > 1
        parenthesized = infix and nested
        if parenthesized:
            print("(", end="", file=file)
        match x:
            case BlockArgument(index=i, type=smt.BoolType()):
                print(
                    ("x", "y", "z", "w", "v", "u", "t", "s")[permutation[i]],
                    end="",
                    file=file,
                )
            case BlockArgument(index=i, type=bv.BitVectorType(width=width)):
                print(
                    ("x", "y", "z", "w", "v", "u", "t", "s")[permutation[i]],
                    end="",
                    file=file,
                )
                print(f"#{width.data}", end="", file=file)
            case OpResult(op=smt.ConstantBoolOp(value=val), index=0):
                print("⊤" if val else "⊥", end="", file=file)
            case OpResult(op=bv.ConstantOp(value=val), index=0):
                width = val.type.width.data
                value = val.value.data
                print(f"{{:0{width}b}}".format(value), end="", file=file)
            case OpResult(op=smt.NotOp(arg=arg), index=0):
                print("¬", end="", file=file)
                Program._pretty_print_value(arg, permutation, True, file=file)
            case OpResult(op=smt.AndOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, permutation, True, file=file)
                print(" ∧ ", end="", file=file)
                Program._pretty_print_value(rhs, permutation, True, file=file)
            case OpResult(op=smt.OrOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, permutation, True, file=file)
                print(" ∨ ", end="", file=file)
                Program._pretty_print_value(rhs, permutation, True, file=file)
            case OpResult(op=smt.ImpliesOp(lhs=lhs, rhs=rhs), index=0):
                Program._pretty_print_value(lhs, permutation, True, file=file)
                print(" → ", end="", file=file)
                Program._pretty_print_value(rhs, permutation, True, file=file)
            case OpResult(op=smt.DistinctOp(lhs=lhs, rhs=rhs), index=0):
                Program._pretty_print_value(lhs, permutation, True, file=file)
                print(" ≠ ", end="", file=file)
                Program._pretty_print_value(rhs, permutation, True, file=file)
            case OpResult(op=smt.EqOp(lhs=lhs, rhs=rhs), index=0):
                Program._pretty_print_value(lhs, permutation, True, file=file)
                print(" = ", end="", file=file)
                Program._pretty_print_value(rhs, permutation, True, file=file)
            case OpResult(op=smt.XOrOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, permutation, True, file=file)
                print(" ⊕ ", end="", file=file)
                Program._pretty_print_value(rhs, permutation, True, file=file)
            case OpResult(
                op=smt.IteOp(cond=cond, true_val=true_val, false_val=false_val), index=0
            ):
                Program._pretty_print_value(cond, permutation, True, file=file)
                print(" ? ", end="", file=file)
                Program._pretty_print_value(true_val, permutation, True, file=file)
                print(" : ", end="", file=file)
                Program._pretty_print_value(false_val, permutation, True, file=file)
            case OpResult(op=bv.AddOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, permutation, True, file=file)
                print(" + ", end="", file=file)
                Program._pretty_print_value(rhs, permutation, True, file=file)
            case OpResult(op=bv.AndOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, permutation, True, file=file)
                print(" & ", end="", file=file)
                Program._pretty_print_value(rhs, permutation, True, file=file)
            case OpResult(op=bv.OrOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, permutation, True, file=file)
                print(" | ", end="", file=file)
                Program._pretty_print_value(rhs, permutation, True, file=file)
            case OpResult(op=bv.MulOp(operands=(lhs, rhs)), index=0):
                Program._pretty_print_value(lhs, permutation, True, file=file)
                print(" * ", end="", file=file)
                Program._pretty_print_value(rhs, permutation, True, file=file)
            case OpResult(op=bv.NotOp(arg=arg), index=0):
                print("~", end="", file=file)
                Program._pretty_print_value(arg, permutation, True, file=file)
            case _:
                raise ValueError(f"Unknown value for pretty print: {x}")
        if parenthesized:
            print(")", end="", file=file)

    def __str__(self) -> str:
        buffer = StringIO()
        Program._pretty_print_value(
            self.ret().arguments[0],
            self._param_permutation,
            False,
            file=buffer,
        )
        return buffer.getvalue()


class RewriteRule:
    _lhs: Program
    _rhs: Program

    def __init__(self, lhs: Program, rhs: Program):
        permuted_rhs = rhs.permute_parameters_to_match(lhs)
        assert permuted_rhs is not None
        self._lhs = lhs
        self._rhs = permuted_rhs

    def __str__(self) -> str:
        return f"{self._lhs} ↭ {self._rhs}"


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
    building_blocks: list[Program],
    illegals: list[Program],
) -> Generator[Program, None, None]:
    # Disabled for now.
    use_building_blocks = len(building_blocks) != 0 and False
    if use_building_blocks:
        building_blocks.sort()
        with open(BUILDING_BLOCKS_FILE, "w") as f:
            size = building_blocks[0].size()
            for program in building_blocks:
                if program.size() != size:
                    size = program.size()
                    f.write("// +++++\n")
                f.write(str(program.module()))
                f.write("\n// -----\n")
            f.write("// +++++\n")

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
            f"--building-blocks={BUILDING_BLOCKS_FILE if use_building_blocks else ''}",
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
) -> bool:
    """
    Check wether two programs are semantically equivalent after permuting the
    arguments of the left program, using Z3. This also checks whether the input
    types match (after permutation and removal of useless parameters).
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
        left.useful_parameters(), right.useful_parameters()
    ):
        if left_type != right_type:
            return False
        arg = builder.insert(smt.DeclareConstOp(left_type)).res
        args_left[left_index] = arg
        args_right[right_index] = arg

    for index, ty in enumerate(func_left.func_type.inputs):
        if args_left[index] is None:
            args_left[index] = builder.insert(smt.DeclareConstOp(ty)).res

    for index, ty in enumerate(func_right.func_type.inputs):
        if args_right[index] is None:
            args_right[index] = builder.insert(smt.DeclareConstOp(ty)).res

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


def is_parameter_useless_z3(program: Program, arg_index: int) -> bool:
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


def find_new_behaviors_in_bucket(
    canonicals: list[Program],
    bucket: Bucket,
) -> tuple[dict[Program, Bucket], list[Bucket]]:
    # Sort programs into actual behavior buckets.
    behaviors: list[Bucket] = []
    for program in bucket:
        for behavior in behaviors:
            if program.is_same_behavior(behavior[0]):
                behavior.append(program)
                break
        else:
            behaviors.append([program])

    # Exclude known behaviors.
    known_behaviors: dict[Program, Bucket] = {}
    new_behaviors: list[Bucket] = []
    for behavior in behaviors:
        for canonical in canonicals:
            if behavior[0].is_same_behavior(canonical):
                known_behaviors[canonical] = behavior
                break
        else:
            new_behaviors.append(behavior)

    return known_behaviors, new_behaviors


def find_new_behaviors(
    buckets: list[Bucket],
    canonicals: list[Program],
) -> tuple[dict[Program, Bucket], list[Bucket]]:
    """
    Returns a `known_behaviors, new_behaviors` pair where `known_behaviors` is a
    map from canonical programs to buckets of new programs with the same
    behavior, and `new_behaviors` is a list of equivalence classes of the
    programs exhibiting a new behavior.
    """

    known_behaviors: dict[Program, Bucket] = {}
    new_behaviors: list[Bucket] = []

    with Pool() as p:
        for i, (known, new) in enumerate(
            p.imap_unordered(partial(find_new_behaviors_in_bucket, canonicals), buckets)
        ):
            print(
                f"\033[2K Finding new behaviors... "
                f"({round(100.0 * i / len(buckets), 1)} %)",
                end="\r",
            )
            known_behaviors.update(known)
            new_behaviors.extend(new)

    return known_behaviors, new_behaviors


def remove_redundant_illegal_subpatterns(
    new_canonicals: list[Program], new_rewrites: dict[Program, Bucket]
) -> tuple[dict[Program, Bucket], int]:
    buffer = StringIO()
    print("module {", file=buffer)
    print("module {", file=buffer)
    for canonical in new_canonicals:
        print(canonical.module(), file=buffer)
    print("}", file=buffer)
    print("module {", file=buffer)
    for programs in new_rewrites.values():
        for program in programs:
            print(program.module(), file=buffer)
    print("}", file=buffer)
    print("}", file=buffer)
    cpp_res = sp.run(
        [REMOVE_REDUNDANT_PATTERNS],
        input=buffer.getvalue(),
        stdout=sp.PIPE,
        stderr=sys.stderr,
        text=True,
    )
    res_lines = cpp_res.stdout.splitlines()

    pruned_rewrites: dict[Program, Bucket] = {
        canonical: [] for canonical in new_rewrites.keys()
    }
    i = 0
    pruned_count = 0
    # Iteration order over a dict is fixed, so we can rely on that.
    for canonical, programs in new_rewrites.items():
        for program in programs:
            if res_lines[i] == "true":
                pruned_count += 1
            else:
                pruned_rewrites[canonical].append(program)
            i += 1
    return pruned_rewrites, pruned_count


def main() -> None:
    global_start = time.time()

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
    rewrites: list[RewriteRule] = []

    try:
        for m in range(args.max_num_ops + 1):
            print(f"\033[1m== Size {m} ==\033[0m")
            step_start = time.time()

            enumerating_start = time.time()
            buckets: dict[Fingerprint, Bucket] = {}
            enumerated_count = 0
            for program in enumerate_programs(
                ctx,
                args.max_num_args,
                m,
                args.bitvector_widths,
                canonicals if m >= 2 else [],
                illegals,
            ):
                enumerated_count += 1
                print(
                    f"\033[2K Enumerating programs... ({enumerated_count})",
                    end="\r",
                )
                fingerprint = program.fingerprint()
                if fingerprint not in buckets:
                    buckets[fingerprint] = []
                buckets[fingerprint].append(program)
            enumerating_time = round(time.time() - enumerating_start, 2)
            print(
                f"\033[2KGenerated {enumerated_count} programs of this size "
                f"in {enumerating_time:.02f} s."
            )

            new_rewrites: dict[Program, Bucket] = {}

            finding_start = time.time()
            known_behaviors, new_behaviors = find_new_behaviors(
                list(buckets.values()), canonicals
            )
            for canonical, programs in known_behaviors.items():
                new_rewrites[canonical] = programs
            finding_time = round(time.time() - finding_start, 2)
            print(
                f"\033[2KFound {len(new_behaviors)} new behaviors, "
                f"exhibited by {sum(len(behavior) for behavior in new_behaviors)} programs "
                f"in {finding_time:.02f} s."
            )

            choosing_start = time.time()
            new_canonicals: list[Program] = []
            for i, behavior in enumerate(new_behaviors):
                print(
                    f"\033[2K Choosing new canonical programs... "
                    f"({i + 1}/{len(new_behaviors)})",
                    end="\r",
                )
                canonical = min(behavior)
                new_canonicals.append(canonical)
                new_rewrites[canonical] = behavior
            canonicals.extend(new_canonicals)
            choosing_time = round(time.time() - choosing_start, 2)
            print(
                f"\033[2KChose {len(new_canonicals)} new canonical programs "
                f"in {choosing_time:.02f} s."
            )

            print(" Removing redundant illegal sub-patterns...", end="\r")
            pruning_start = time.time()
            pruned_rewrites, pruned_count = remove_redundant_illegal_subpatterns(
                new_canonicals, new_rewrites
            )
            for new_illegals in pruned_rewrites.values():
                illegals.extend(new_illegals)
            rewrites.extend(
                RewriteRule(program, canonical)
                for canonical, bucket in pruned_rewrites.items()
                for program in bucket
            )
            pruning_time = round(time.time() - pruning_start, 2)
            print(
                f"\033[2KRemoved {pruned_count} redundant illegal sub-patterns "
                f"in {pruning_time:.02f} s."
            )

            step_end = time.time()
            print(f"Finished step in {round(step_end - step_start, 2):.02f} s.")
            print(
                f"We now have a total of {len(canonicals)} behaviors "
                f"and {len(illegals)} illegal sub-patterns."
            )

        if args.out_canonicals != "":
            with open(args.out_canonicals, "w", encoding="UTF-8") as f:
                for program in canonicals:
                    f.write(str(program.module()))
                    f.write("\n// -----\n")

        if args.out_rewrites != "":
            print("Outputing rewrites is not supported yet")
            # TODO: Take `Program._param_permutation` into account.
            # with open(args.out_rewrites, "w", encoding="UTF-8") as f:
            #     for canonical, programs in rewrites.items():
            #         for program in programs:
            #             f.write(str(program.module()))
            #             f.write("\n// =====\n")
            #             f.write(str(program.module()))
            #             f.write("\n// -----\n")

        if args.summarize_canonicals:
            print(f"\033[1m== Summary (canonical programs) ==\033[0m")
            for program in canonicals:
                print(program)

        if args.summarize_rewrites:
            print(f"\033[1m== Summary (rewrite rules) ==\033[0m")
            for rewrite in rewrites:
                print(rewrite)

    except BrokenPipeError:
        # The enumerator has terminated
        pass
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        exit(1)
    finally:
        global_end = time.time()
        print(f"Total time: {round(global_end - global_start):.02f} s.")


if __name__ == "__main__":
    main()
