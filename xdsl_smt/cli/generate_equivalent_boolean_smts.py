#!/usr/bin/env python3

import argparse
import copy
import itertools
import subprocess as sp
import sys
import time
import z3  # pyright: ignore[reportMissingTypeStubs]
from dataclasses import dataclass, fields
from enum import Enum, auto
from functools import cache, partial
from io import StringIO
from multiprocessing import Pool
from typing import (
    IO,
    Any,
    Callable,
    Generic,
    Iterable,
    Sequence,
    TypeVar,
    cast,
)

from xdsl.context import Context
from xdsl.ir import Attribute, Region
from xdsl.ir.core import Block, BlockArgument, Operation, OpResult, SSAValue
from xdsl.parser import Parser
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.builder import Builder
from xdsl_smt.utils.pretty_print import pretty_print_value

import xdsl_smt.dialects.synth_dialect as synth
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_bitvector_dialect as bv
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.dialects import pdl
from xdsl.dialects.builtin import (
    Builtin,
    IntAttr,
    IntegerAttr,
    ModuleOp,
    StringAttr,
)
from xdsl.dialects.func import Func, FuncOp, ReturnOp
from xdsl.dialects.builtin import FunctionType

from xdsl_smt.cli.xdsl_smt_run import build_interpreter, interpret
from xdsl_smt.traits.smt_printer import print_to_smtlib


MLIR_ENUMERATE = "./mlir-fuzz/build/bin/mlir-enumerate"
REMOVE_REDUNDANT_PATTERNS = "./mlir-fuzz/build/bin/remove-redundant-patterns"
SMT_MLIR = "./mlir-fuzz/dialects/smt.mlir"
EXCLUDE_SUBPATTERNS_FILE = f"/tmp/exclude-subpatterns-{time.time()}.mlir"
BUILDING_BLOCKS_FILE = f"/tmp/building-blocks-{time.time()}.mlir"


T = TypeVar("T")


class FrozenMultiset(Generic[T]):
    __slots__ = ("_contents",)

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


Result = tuple[Any, ...]
Image = tuple[Result, ...]


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """
    A value that can be computed from a program, and highly depends on its
    behavior.

    The fingerprints of two semantically equivalent programs are guaranteed to
    compare equal. Furthermore, if two programs are semantically equivalent
    after removing inputs that don't affect the output, and optionally permuting
    their parameters, their fingerprints are guaranteed to compare equal as
    well.
    """

    _useful_input_types: FrozenMultiset[Attribute]
    _output_types: tuple[Attribute, ...]
    _images: FrozenMultiset[Image]


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


def func_to_pdl(
    func: FuncOp, *, arguments: Sequence[SSAValue | None] | None = None
) -> tuple[Region, Callable[[int], SSAValue], SSAValue | None, tuple[SSAValue, ...]]:
    """
    Creates a region containing PDL instructions corresponding to this
    program. Returns a containing `Region`, together with the values of the
    inputs, the root operation (i.e., the operation the first returned value
    comes from, if it exists), and the returned values.
    """

    if arguments is not None:
        assert len(arguments) == len(func.function_type.inputs)

    body = Region(Block())
    builder = Builder(InsertPoint.at_end(body.block))

    @cache
    def get_type(ty: Attribute) -> SSAValue:
        return builder.insert(pdl.TypeOp(ty)).results[0]

    @cache
    def get_attribute(attr: Attribute) -> SSAValue:
        return builder.insert(pdl.AttributeOp(attr)).results[0]

    @cache
    def get_argument(arg: int) -> SSAValue:
        if arguments is not None:
            if (value := arguments[arg]) is not None:
                return value
        return builder.insert(pdl.OperandOp(get_type(func.args[arg].type))).results[0]

    ops: dict[Operation, SSAValue] = {}

    def get_value(value: SSAValue) -> SSAValue:
        match value:
            case BlockArgument(index=k):
                return get_argument(k)
            case OpResult(op=op, index=i):
                return builder.insert(pdl.ResultOp(i, ops[op])).results[0]
            case x:
                raise ValueError(f"Unknown value: {x}")

    [*operations, ret] = list(func.body.ops)
    for op in operations:
        pattern = builder.insert(
            pdl.OperationOp(
                op.name,
                [StringAttr(s) for s in op.properties.keys()],
                [get_value(operand) for operand in op.operands],
                # Iteration order on dict is consistent betwen `.keys()` and
                # `.values()`.
                [get_attribute(prop) for prop in op.properties.values()],
                [get_type(ty) for ty in op.result_types],
            )
        )
        ops[op] = pattern.results[0]

    assert isinstance(ret, ReturnOp)
    assert len(ret.operands) == 1
    root = ret.operands[0]

    return (
        body,
        get_argument,
        ops[root.op] if isinstance(root, OpResult) else None,
        tuple(get_value(res) for res in ret.operands),
    )


class Program:
    __slots__ = (
        "_module",
        "_size",
        "_cost",
        "_input_cardinalities",
        "_base_image",
        "_fingerprint",
        "_is_basic",
        "_param_permutation",
        "_useless_param_count",
    )

    _module: ModuleOp
    _size: int
    _cost: int
    _input_cardinalities: tuple[int, ...]
    """
    The cardinality of each input, before applying the permutation.
    """
    _base_image: Image
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
            case OpResult(op=op):
                if len(op.operands) == 0:
                    return 0
                return 1 + sum(
                    Program._formula_size(operand) for operand in op.operands
                )
            case x:
                raise ValueError(f"Unknown value: {x}")

    @staticmethod
    def _operation_cost(op: Operation) -> int:
        match op:
            case ReturnOp():
                return 0
            case (
                bv.MulOp()
                | bv.URemOp()
                | bv.SRemOp()
                | bv.SModOp()
                | bv.UDivOp()
                | bv.SDivOp()
            ):
                return 4
            case op:
                if len(op.operands) == 0:
                    return 0
                return 1

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
        """
        Returns a new version of this program, with the useful parameters
        permuted according to the specified permutation. Useless parameters are
        unaffected, and stay at the end of the parameter permutation.
        """
        arity = self.useful_arity()
        assert len(permutation) == arity
        permuted = copy.copy(self)
        permuted._param_permutation = list(permuted._param_permutation)
        permuted._param_permutation[:arity] = [
            permuted._param_permutation[i] for i in permutation
        ]
        return permuted

    def _parameter_permutations(self) -> Iterable["Program"]:
        """
        Returns an iterator over all versions of this programs with useful
        permuted useful parameters. Useless parameters are guaranteed to always
        appear at the end of the permutations.
        """
        for permutation in itertools.permutations(range(self.useful_arity())):
            yield self.permute_useful_parameters(permutation)

    def image(self) -> Image:
        """
        Returns the image of this program, taking the parameter permutation
        into account.
        """
        assert self._input_cardinalities is not None
        assert self._base_image is not None
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
            self._base_image[indices[reverse_permute(piid, self._param_permutation)]]
            for piid in permuted_input_ids
        )

    def __init__(self, module: ModuleOp):
        self._module = module

        self._size = sum(
            Program._formula_size(argument) for argument in self.ret().arguments
        )
        self._cost = sum(Program._operation_cost(op) for op in self.func().body.ops)

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
        self._base_image = tuple(
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
            permuted.image() for permuted in self._parameter_permutations()
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
        """
        Returns the number of parameters this program accepts, including useless
        parameters.
        """
        return len(self.func().function_type.inputs)

    def useful_arity(self) -> int:
        """Returns the number of useful parameters for this program."""
        return self.arity() - self._useless_param_count

    def size(self) -> int:
        return self._size

    def cost(self) -> int:
        """
        Returns the cost of this program, according to a very basic cost model.
        """
        return self._cost

    def fingerprint(self) -> Fingerprint:
        return self._fingerprint

    def is_basic(self) -> bool:
        """
        Whether the whole behavior of this program is encapsulated in its
        fingerprint. If two programs have are basic, they are equivalent if, and
        only if, their fingerprints compare equal.
        """
        return self._is_basic

    def parameter(self, index: int) -> int:
        """
        Returns the index of the passed parameter in the underlying function.
        """
        return self._param_permutation[index]

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
    def _compare_values_lexicographically(
        left: SSAValue,
        right: SSAValue,
        left_params: Permutation,
        right_params: Permutation,
    ) -> int:
        match left, right:
            # Order parameters.
            case BlockArgument(index=i), BlockArgument(index=j):
                return left_params[i] - right_params[j]
            # Favor operations over arguments.
            case OpResult(), BlockArgument():
                return -1
            case BlockArgument(), OpResult():
                return 1
            # Order operations.
            case OpResult(op=lop, index=i), OpResult(op=rop, index=j):
                # Favor operations with smaller arity.
                if len(lop.operands) != len(rop.operands):
                    return len(lop.operands) - len(rop.operands)
                # Choose an arbitrary result if they are different.
                if not isinstance(lop, type(rop)):
                    return hash(lop.name) - hash(rop.name)
                if lop.properties != rop.properties:
                    return hash(frozenset(lop.properties.items())) - hash(
                        frozenset(rop.properties.items())
                    )
                if i != j:
                    return i - j
                # Compare operands as a last resort.
                for lo, ro in zip(lop.operands, rop.operands, strict=True):
                    c = Program._compare_values_lexicographically(
                        lo, ro, left_params, right_params
                    )
                    if c != 0:
                        return c
                return 0
            case l, r:
                raise ValueError(f"Unknown value: {l} or {r}")

    def _compare_lexicographically(self, other: "Program") -> int:
        # Favor smaller programs.
        if self.size() < other.size():
            return -1
        if self.size() > other.size():
            return 1
        # Favor programs with fewer useless parameters.
        if self._useless_param_count < other._useless_param_count:
            return -1
        if self._useless_param_count > other._useless_param_count:
            return 1
        # Favor programs with fewer parameters overall.
        if self.arity() < other.arity():
            return -1
        if self.arity() > other.arity():
            return 1
        # Favor programs that return less values.
        self_outs = self.ret().arguments
        other_outs = other.ret().arguments
        if len(self_outs) < len(other_outs):
            return -1
        if len(self_outs) > len(other_outs):
            return 1
        # Compare formula trees for each return value.
        for self_out, other_out in zip(self_outs, other_outs, strict=True):
            c = Program._compare_values_lexicographically(
                self_out,
                other_out,
                self._param_permutation,
                other._param_permutation,
            )
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

    def computes_same_function(self, other: "Program") -> bool:
        """
        Tests whether two programs are logically equivalent in the usual sense.
        """

        if self.image() != other.image():
            return False

        if self.is_basic() and other.is_basic():
            return True

        return is_same_behavior_with_z3(self, other)

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
            if permuted.computes_same_function(other):
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
            if permuted.image() == other.image():
                if (
                    self.is_basic()
                    and other.is_basic()
                    or is_same_behavior_with_z3(permuted, other)
                ):
                    return permuted

        return None

    def to_pdl(
        self, *, arguments: Sequence[SSAValue | None] | None = None
    ) -> tuple[
        Region, Callable[[int], SSAValue], SSAValue | None, tuple[SSAValue, ...]
    ]:
        """
        Creates a region containing PDL instructions corresponding to this
        program. Returns a containing `Region`, together with the values of the
        inputs, the root operation (i.e., the operation the first returned value
        comes from, if it exists), and the returned values.
        """
        return func_to_pdl(self.func(), arguments=arguments)

    def pdl_pattern(self) -> ModuleOp:
        """Creates a PDL pattern from this program."""

        body, _, root, _ = self.to_pdl()
        body.block.add_op(pdl.RewriteOp(root))
        pattern = pdl.PatternOp(1, None, body)

        return ModuleOp([pattern])

    def __str__(self) -> str:
        buffer = StringIO()
        available_names = ("x", "y", "z", "w", "v", "u", "t", "s")
        names: dict[SSAValue, str] = {}

        for index, arg in zip(self._param_permutation, self.func().args):
            names[arg] = available_names[index]
            if isinstance(arg.type, bv.BitVectorType):
                names[arg] += f"#{arg.type.width.data}"

        pretty_print_value(
            self.ret().arguments[0],
            False,
            names,
            file=buffer,
        )
        return buffer.getvalue()


def pretty_print_func(
    func: FuncOp,
    names: Sequence[str] = ("x", "y", "z", "w", "v", "u", "t", "s"),
    *,
    file: IO[str] = sys.stdout,
):
    val_names: dict[SSAValue, str] = {}
    for index, arg in enumerate(func.args):
        val_names[arg] = names[index]
        if isinstance(arg.type, bv.BitVectorType):
            val_names[arg] += f"#{arg.type.width.data}"
    return_op = func.get_return_op()
    assert return_op is not None
    pretty_print_value(
        return_op.arguments[0],
        False,
        val_names,
        file=file,
    )


class RewriteRule:
    __slots__ = ("_lhs", "_rhs")

    _lhs: Program
    _rhs: Program

    def __init__(self, lhs: Program, rhs: Program):
        permuted_rhs = rhs.permute_parameters_to_match(lhs)
        assert permuted_rhs is not None
        self._lhs = lhs
        self._rhs = permuted_rhs

    def to_pdl(self) -> pdl.PatternOp:
        """Expresses this rewrite rule as a PDL pattern and rewrite."""

        lhs, get_arg, left_root, _ = self._lhs.to_pdl()
        assert left_root is not None
        pattern = pdl.PatternOp(1, None, lhs)

        # Unify LHS and RHS arguments.
        arguments: list[SSAValue | None] = [None] * self._rhs.arity()
        for i, (k, _) in enumerate(self._rhs.useful_parameters()):
            arguments[k] = get_arg(self._lhs.parameter(i))
        rhs, _, _, right_res = self._rhs.to_pdl(arguments=arguments)
        rhs.block.add_op(pdl.ReplaceOp(left_root, None, right_res))

        pattern.body.block.add_op(pdl.RewriteOp(left_root, rhs))

        return pattern

    def __str__(self) -> str:
        return f"{self._lhs} ⇝ {self._rhs}"


Bucket = list[Program]


class EnumerationOrder(Enum):
    SIZE = auto()
    COST = auto()

    @classmethod
    def parse(cls, arg: str):
        if arg == "size":
            return cls.SIZE
        if arg == "cost":
            return cls.COST
        raise ValueError("Invalid enumeration order: {arg!r}")

    def phase(self, program: "Program") -> int:
        match self:
            case EnumerationOrder.SIZE:
                return program.size()
            case EnumerationOrder.COST:
                return program.cost()

    def __str__(self):
        return self.name.lower()


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "--max-num-args",
        type=int,
        help="maximum number of arguments in the generated MLIR programs",
        default=999999999,
    )
    arg_parser.add_argument(
        "--phases",
        type=int,
        help="the number of phases",
    )
    arg_parser.add_argument(
        "--bitvector-widths",
        type=str,
        help="a list of comma-separated bitwidths",
        default="4",
    )
    arg_parser.add_argument(
        "--enumeration-order",
        type=EnumerationOrder.parse,
        choices=tuple(EnumerationOrder),
        help="the order in which to enumerate programs",
        default=EnumerationOrder.SIZE,
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

    arg_parser.add_argument(
        "--dialect",
        dest="dialect",
        default=SMT_MLIR,
        help="The IRDL file describing the dialect to use for enumeration",
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


def enumerate_programs(
    max_num_args: int,
    num_ops: int,
    bv_widths: str,
    building_blocks: list[Program],
    illegals: list[FuncOp],
    dialect_path: str,
    additional_options: Sequence[str] = (),
) -> Iterable[str]:
    # Disabled for now.
    use_building_blocks = len(building_blocks) != 0
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
            body, _, root, _ = func_to_pdl(program)
            body.block.add_op(pdl.RewriteOp(root))
            pattern = pdl.PatternOp(1, None, body)
            pdl_module = ModuleOp([pattern])

            f.write(str(pdl_module))
            f.write("\n// -----\n")

    enumerator = sp.Popen(
        [
            MLIR_ENUMERATE,
            dialect_path,
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
            *additional_options,
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

        yield source


def parse_program(source: str) -> Program:
    ctx = Context()
    ctx.allow_unregistered = True
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(SMTDialect)
    ctx.load_dialect(SMTBitVectorDialect)
    ctx.load_dialect(SMTUtilsDialect)
    ctx.load_dialect(synth.SynthDialect)

    return Program(Parser(ctx, source).parse_module(True))


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


def clone_func_to_smt_func_with_constants(func: FuncOp) -> smt.DefineFunOp:
    """
    Convert a `func.func` to an `smt.define_fun` operation.
    Additionally, move `synth.constant` operations to arguments.
    Do not mutate the original function.
    """
    new_region = func.body.clone()
    new_block = new_region.block

    # Replace the `func.return` with an `smt.return` operation.
    func_return = new_block.last_op
    assert isinstance(func_return, ReturnOp)
    rewriter = Rewriter()
    rewriter.insert_op(
        smt.ReturnOp(func_return.arguments), InsertPoint.before(func_return)
    )
    rewriter.erase_op(func_return)

    # Move all `synth.constant` operations to arguments.
    for op in tuple(new_block.walk()):
        if isinstance(op, synth.ConstantOp):
            new_arg = new_block.insert_arg(op.res.type, len(new_block.args))
            rewriter.replace_op(op, [], [new_arg])

    return smt.DefineFunOp(new_region)


def run_module_through_smtlib(module: ModuleOp) -> Any:
    smtlib_program = StringIO()
    print_to_smtlib(module, smtlib_program)

    # Parse the SMT-LIB program and run it through the Z3 solver.
    solver = z3.Solver()
    # Set the timeout
    solver.set("timeout", 25000)  # pyright: ignore[reportUnknownMemberType]
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
        return z3.unknown
        raise Exception()
    return result


def inline_single_result_func(
    func: FuncOp, args: Sequence[SSAValue], insert_point: InsertPoint
) -> SSAValue:
    """
    Inline a single-result function at the current location.
    """
    assert len(func.function_type.outputs) == 1, "Function must have a single output."
    assert len(func.body.blocks) == 1, "Function must have a single block."

    block_copy = func.body.clone().block
    return_op = block_copy.last_op
    assert isinstance(return_op, ReturnOp), "Function must end with a return operation."
    return_value = return_op.operands[0]
    block_copy.erase_op(return_op)

    if return_value in block_copy.args:
        assert isinstance(return_value, BlockArgument)
        return_value = args[return_value.index]

    Rewriter.inline_block(block_copy, insert_point, args)
    return return_value


def combine_funcs_with_synth_constants(funcs: Sequence[FuncOp]) -> FuncOp:
    """
    Combine multiple `func.func` operations into a single one, using
    synth.constant operations to choose which function to call.
    """
    assert len(funcs) > 0, "At least one function is required."

    funcs = list(funcs)
    while len(funcs) > 1:
        left = funcs.pop().clone()
        right = funcs.pop().clone()

        assert left.function_type == right.function_type

        merged_func = FuncOp(left.name, left.function_type)
        insert_point = InsertPoint.at_end(merged_func.body.block)
        left_val = inline_single_result_func(left, merged_func.args, insert_point)
        right_val = inline_single_result_func(right, merged_func.args, insert_point)
        builder = Builder(insert_point)

        # Create a synth.constant to choose which function to call.
        cst = builder.insert(synth.ConstantOp(smt.BoolType()))
        # Create a conditional operation to choose the function to call, and return
        # the result.
        cond = builder.insert(smt.IteOp(cst.res, left_val, right_val)).res
        builder.insert(ReturnOp(cond))

        # Add the merged function to the list of functions.
        funcs.append(merged_func)

    return funcs[0].clone()


def is_range_subset_of_list_with_z3(
    left: FuncOp,
    right: Sequence[FuncOp],
):
    """
    Check wether the ranges of values that the `left` program can reach by assigning
    constants to `xdsl.smt.synth` is a subset of the range of values that the `right`
    programs can reach. This is done using Z3.
    """

    merged_right = combine_funcs_with_synth_constants(right)
    return is_range_subset_with_z3(left, merged_right)


def is_range_subset_with_z3(
    left: FuncOp,
    right: FuncOp,
):
    """
    Check wether the ranges of values that the `left` program can reach by assigning
    constants to `xdsl.smt.synth` is a sbuset of the range of values that the `right`
    program can reach. This is done using Z3.
    """

    module = ModuleOp([])
    builder = Builder(InsertPoint.at_end(module.body.block))

    # Clone both functions into a new module.
    func_left = clone_func_to_smt_func_with_constants(left)
    func_right = clone_func_to_smt_func_with_constants(right)
    builder.insert(func_left)
    builder.insert(func_right)

    toplevel_val: SSAValue | None = None

    # Create the lhs constant foralls.
    lhs_cst_types = func_left.func_type.inputs.data[
        len(left.function_type.inputs.data) :
    ]
    if lhs_cst_types:
        forall_cst = builder.insert(smt.ForallOp.from_variables(lhs_cst_types))
        lhs_cst_args = forall_cst.body.block.args
        builder.insertion_point = InsertPoint.at_end(forall_cst.body.block)
        toplevel_val = forall_cst.res
    else:
        lhs_cst_args = ()

    # Create the rhs constant exists.
    rhs_cst_types = func_right.func_type.inputs.data[
        len(right.function_type.inputs.data) :
    ]
    if rhs_cst_types:
        exists_cst = builder.insert(smt.ExistsOp.from_variables(rhs_cst_types))
        rhs_cst_args = exists_cst.body.block.args
        if toplevel_val is not None:
            builder.insert(smt.YieldOp(exists_cst.res))
        else:
            toplevel_val = exists_cst.res
        builder.insertion_point = InsertPoint.at_end(exists_cst.body.block)
    else:
        rhs_cst_args = ()

    # Create the variable forall and the assert.
    if left.function_type.inputs.data:
        forall = builder.insert(
            smt.ForallOp.from_variables(left.function_type.inputs.data)
        )
        var_args = forall.body.block.args
        if toplevel_val is not None:
            builder.insert(smt.YieldOp(forall.res))
        else:
            toplevel_val = forall.res
        builder.insertion_point = InsertPoint.at_end(forall.body.block)
    else:
        var_args = ()

    # Call both functions and check for equality.
    args_left = (*var_args, *lhs_cst_args)
    args_right = (*var_args, *rhs_cst_args)
    call_left = builder.insert(smt.CallOp(func_left.ret, args_left)).res
    call_right = builder.insert(smt.CallOp(func_right.ret, args_right)).res
    check = builder.insert(smt.EqOp(call_left[0], call_right[0])).res
    if toplevel_val is not None:
        builder.insert(smt.YieldOp(check))
    else:
        toplevel_val = check

    builder.insertion_point = InsertPoint.at_end(module.body.block)
    builder.insert(smt.AssertOp(toplevel_val))

    return z3.sat == run_module_through_smtlib(
        module
    )  # pyright: ignore[reportUnknownVariableType]


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
    canonicals: dict[Fingerprint, list[Program]],
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
        for canonical in canonicals.get(behavior[0].fingerprint(), []):
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

    canonicals_dict: dict[Fingerprint, list[Program]] = {}
    for canonical in canonicals:
        canonicals_dict.setdefault(canonical.fingerprint(), []).append(canonical)

    known_behaviors: dict[Program, Bucket] = dict[Program, Bucket]()
    new_behaviors: list[Bucket] = []

    with Pool() as p:
        for i, (known, new) in enumerate(
            p.imap_unordered(
                partial(find_new_behaviors_in_bucket, canonicals_dict), buckets
            )
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


@dataclass(frozen=True, slots=True)
class BucketStat:
    phase: int
    bck_cnt: int
    avg_sz: float | None
    min_sz: int | None
    med_sz: int | None
    max_sz: int | None
    exp_sz: float | None
    """Expected value of the size of a random program's bucket."""

    @classmethod
    def from_buckets(cls, phase: int, buckets: Iterable[Bucket]):
        bucket_sizes = sorted(len(bucket) for bucket in buckets)
        n = len(bucket_sizes)
        return cls(
            phase,
            n,
            round(sum(bucket_sizes) / n, 2) if n != 0 else None,
            bucket_sizes[0] if n != 0 else None,
            bucket_sizes[n // 2] if n != 0 else None,
            bucket_sizes[-1] if n != 0 else None,
            (
                round(
                    sum(size * size for size in bucket_sizes)
                    / sum(size for size in bucket_sizes),
                    2,
                )
                if n != 0
                else None
            ),
        )

    @classmethod
    def headers(cls) -> str:
        return "\t".join(f.name for f in fields(cls))

    def _value(self, name: str) -> str:
        x = getattr(self, name)
        if x is None:
            return "N/A"
        return str(x)

    def __str__(self) -> str:
        return "\t".join(self._value(f.name) for f in fields(type(self)))


def main() -> None:
    global_start = time.time()

    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    canonicals: list[Program] = []
    illegals: list[Program] = []
    rewrites: list[RewriteRule] = []
    bucket_stats: list[BucketStat] = []

    try:
        for phase in range(args.phases + 1):
            phase_start = time.time()

            print(f"\033[1m== Phase {phase} (size at most {phase}) ==\033[0m")

            enumerating_start = time.time()
            buckets: dict[Fingerprint, Bucket] = {}
            enumerated_count = 0
            with Pool() as p:
                for program in p.imap_unordered(
                    parse_program,
                    enumerate_programs(
                        args.max_num_args,
                        phase,
                        args.bitvector_widths,
                        canonicals if phase >= 2 else [],
                        [illegal.func() for illegal in illegals],
                        args.dialect,
                    ),
                ):
                    if args.enumeration_order.phase(program) != phase:
                        continue
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
            bucket_stats.append(BucketStat.from_buckets(phase, buckets.values()))

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

            phase_end = time.time()
            print(f"Finished phase in {round(phase_end - phase_start, 2):.02f} s.")
            print(
                f"We now have a total of {len(canonicals)} behaviors "
                f"and {len(illegals)} illegal sub-patterns."
            )

        print(f"\033[1m== Results ==\033[0m")
        print("Bucket stats:")
        print(BucketStat.headers())
        for bucket_stat in bucket_stats:
            print(bucket_stat)

        if args.out_canonicals != "":
            with open(args.out_canonicals, "w", encoding="UTF-8") as f:
                for program in canonicals:
                    f.write(str(program.module()))
                    f.write("\n// -----\n")

        if args.out_rewrites != "":
            module = ModuleOp([rewrite.to_pdl() for rewrite in rewrites])
            with open(args.out_rewrites, "w", encoding="UTF-8") as f:
                f.write(str(module))
                f.write("\n")

        if args.summarize_canonicals:
            print(f"\033[1m== Canonical programs ({len(canonicals)}) ==\033[0m")
            for program in canonicals:
                print(program)

        if args.summarize_rewrites:
            print(f"\033[1m== Rewrite rules ({len(rewrites)}) ==\033[0m")
            for rewrite in rewrites:
                print(rewrite)

        if False:
            ctx = Context()
            ctx.allow_unregistered = True
            ctx.load_dialect(Builtin)
            ctx.load_dialect(Func)
            ctx.load_dialect(SMTDialect)
            ctx.load_dialect(SMTBitVectorDialect)
            ctx.load_dialect(SMTUtilsDialect)
            ctx.load_dialect(synth.SynthDialect)

            cst_canonicals = list[FuncOp]()
            cst_illegals = list[FuncOp]()
            for phase in range(args.phases + 1):
                programs = list[FuncOp]()
                print("Enumerating programs in phase", phase)
                for program_idx, program_str in enumerate(
                    enumerate_programs(
                        args.max_num_args,
                        phase,
                        args.bitvector_widths,
                        [],
                        [illegal.func() for illegal in illegals] + cst_illegals,
                        args.dialect,
                        ["--constant-kind=synth"],
                    )
                ):
                    module = Parser(ctx, program_str).parse_module(True)
                    func = module.body.block.first_op
                    assert isinstance(func, FuncOp)
                    func.detach()
                    print("Enumerated", program_idx + 1, "programs", end="\r")

                    should_skip = False
                    for canonical in cst_canonicals:
                        if func.is_structurally_equivalent(canonical):
                            should_skip = True
                            break
                    for illegal in cst_illegals:
                        if func.is_structurally_equivalent(illegal):
                            should_skip = True
                            break
                    if not should_skip:
                        programs.append(func)

                # Group canonical programs by their function type, and merge them using
                # synth.constant.
                grouped_cst_canonicals: dict[FunctionType, list[FuncOp]] = {}
                for canonical in cst_canonicals:
                    grouped_cst_canonicals.setdefault(
                        canonical.function_type, []
                    ).append(canonical)
                merged_cst_canonicals: dict[FunctionType, FuncOp] = {}
                for function_type, ops in grouped_cst_canonicals.items():
                    merged_cst_canonicals[
                        function_type
                    ] = combine_funcs_with_synth_constants(ops)
                    merged_cst_canonicals[function_type].verify()

                new_possible_canonicals = list[FuncOp]()
                if False:
                    for program_idx, program in enumerate(programs):
                        print(
                            f"\033[2K Checking program {program_idx + 1}/{len(programs)}",
                            end="\r",
                        )
                        if program.function_type in merged_cst_canonicals:
                            cst_canonical = merged_cst_canonicals[program.function_type]
                            if is_range_subset_with_z3(program, cst_canonical):
                                print("Found illegal pattern:                   ")
                                pretty_print_func(program)
                                print("\r")
                                cst_illegals.append(program)
                                continue
                        cst_canonicals.append(program)
                else:
                    for program_idx, program in enumerate(programs):
                        for canonical_idx, canonical in enumerate(cst_canonicals):
                            print(
                                f"\033[2K Checking program {program_idx + 1}/{len(programs)} against old programs {canonical_idx + 1}/{len(cst_canonicals)}",
                                end="\r",
                            )
                            if program.function_type != canonical.function_type:
                                continue

                            if is_range_subset_with_z3(program, canonical):
                                print("Found illegal pattern:")
                                pretty_print_func(program)
                                print("which is a subset of:")
                                pretty_print_func(canonical)
                                cst_illegals.append(program)
                                break
                        else:
                            new_possible_canonicals.append(program)
                print()

                is_illegal_mask: list[bool] = [False] * len(new_possible_canonicals)
                for lhs_idx, lhs in enumerate(new_possible_canonicals):
                    for rhs_idx, rhs in enumerate(new_possible_canonicals):
                        print(
                            f"\033[2K Checking program for canonical {lhs_idx + 1}/{len(new_possible_canonicals)} against {rhs_idx + 1}/{len(new_possible_canonicals)}",
                            end="\r",
                        )

                        if is_illegal_mask[rhs_idx]:
                            continue
                        if lhs.function_type != rhs.function_type:
                            continue
                        if lhs is rhs:
                            continue
                        if is_range_subset_with_z3(lhs, rhs):
                            cst_illegals.append(lhs)
                            is_illegal_mask[lhs_idx] = True
                            print("Found illegal pattern:")
                            print(lhs)
                            print("which is a subset of:")
                            print(rhs)
                            break
                    else:
                        cst_canonicals.append(lhs)
                print(f"== At step {phase} ==")
                print("number of canonicals", len(cst_canonicals))
                print("number of illegals", len(cst_illegals))

                # Write the canonicals and illegals to files.
                if args.out_canonicals != "":
                    with open(args.out_canonicals, "w", encoding="UTF-8") as f:
                        for program in cst_canonicals:
                            f.write(str(program))
                            f.write("\n// -----\n")

                        f.write("\n\n\n// +++++ Illegals +++++ \n\n\n")

                        for program in cst_illegals:
                            f.write(str(program))
                            f.write("\n// -----\n")

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
