#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import subprocess as sp
import sys
import time
import z3  # pyright: ignore[reportMissingTypeStubs]
from dataclasses import dataclass, fields
from enum import Enum, auto
from functools import partial
from io import StringIO
from multiprocessing import Pool
from typing import (
    IO,
    Any,
    Iterable,
    Sequence,
    TypeVar,
    cast,
)

from xdsl.context import Context
from xdsl.ir import Region, Block
from xdsl.ir.core import Operation, SSAValue
from xdsl.parser import Parser
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.builder import Builder
from xdsl_smt.superoptimization.pattern import (
    Pattern,
    UnorderedFingerprint,
    OrderedPattern,
)
from xdsl_smt.utils.pretty_print import pretty_print_value
from xdsl_smt.utils.run_with_smt_solver import run_module_through_smtlib
from xdsl_smt.utils.inlining import inline_single_result_func
from xdsl_smt.utils.pdl import func_to_pdl
from xdsl_smt.superoptimization.program_enumeration import enumerate_programs

import xdsl_smt.dialects.synth_dialect as synth
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_bitvector_dialect as bv
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.dialects import pdl
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import (
    Builtin,
    ModuleOp,
)
from xdsl.dialects.func import Func, FuncOp, ReturnOp
from xdsl.dialects.builtin import FunctionType

from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import (
    load_vanilla_semantics_using_control_flow_dialects,
)
from xdsl_smt.passes.smt_expand import SMTExpand
from xdsl_smt.passes.lower_to_smt.lower_to_smt import LowerToSMTPass
from xdsl_smt.passes.lower_effects import LowerEffectPass

sys.setrecursionlimit(100000)

MLIR_ENUMERATE = "./mlir-fuzz/build/bin/mlir-enumerate"
REMOVE_REDUNDANT_PATTERNS = "./mlir-fuzz/build/bin/remove-redundant-patterns"
SMT_MLIR = "./mlir-fuzz/dialects/smt.mlir"
EXCLUDE_SUBPATTERNS_FILE = f"/tmp/exclude-subpatterns-{time.time()}.mlir"
BUILDING_BLOCKS_FILE = f"/tmp/building-blocks-{time.time()}.mlir"


T = TypeVar("T")

Result = tuple[Any, ...]
Image = tuple[Result, ...]


Permutation = tuple[int, ...]


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


def operation_cost(op: Operation) -> int:
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

    _lhs: OrderedPattern
    _rhs: OrderedPattern

    def __init__(self, lhs: Pattern, rhs: Pattern):
        ordered_lhs = next(iter(lhs.ordered_patterns()))
        ordered_rhs = rhs.permute_parameters_to_match(ordered_lhs)
        assert ordered_rhs is not None
        self._lhs = ordered_lhs
        self._rhs = ordered_rhs

    def to_pdl(self) -> pdl.PatternOp:
        """Expresses this rewrite rule as a PDL pattern and rewrite."""

        lhs, args, left_root, _ = func_to_pdl(self._lhs.func)
        assert left_root is not None
        pattern = pdl.PatternOp(1, None, lhs)

        # Unify LHS and RHS arguments.
        arguments: list[SSAValue | None] = [None] * self._rhs.arity
        for i, (k, _) in enumerate(self._rhs.useful_semantics_parameters()):
            arguments[k] = args[self._lhs.permutation[i]]
        rhs, _, _, right_res = func_to_pdl(self._rhs.func, arguments=arguments)
        rhs.block.add_op(pdl.ReplaceOp(left_root, None, right_res))

        pattern.body.block.add_op(pdl.RewriteOp(left_root, rhs))

        return pattern

    def __str__(self) -> str:
        return f"{self._lhs} ⇝ {self._rhs}"


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

    def phase(self, program: Pattern) -> int:
        match self:
            case EnumerationOrder.SIZE:
                return program.size
            case EnumerationOrder.COST:
                return sum(operation_cost(op) for op in program.func.body.ops)

    def __str__(self):
        return self.name.lower()


class Configuration(Enum):
    """
    Different configurations depending on the kind of input programs.
    For instance, LLVM needs to be lowered to SMT first before being able
    to reason about it.
    """

    SMT = "smt"
    ARITH = "arith"


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

    arg_parser.add_argument(
        "--configuration",
        dest="configuration",
        type=Configuration,
        choices=tuple(Configuration),
        default=Configuration.SMT,
    )


def parse_program(configuration: Configuration, source: str) -> Pattern:
    ctx = Context()
    ctx.allow_unregistered = True
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(SMTDialect)
    ctx.load_dialect(SMTBitVectorDialect)
    ctx.load_dialect(SMTUtilsDialect)
    ctx.load_dialect(synth.SynthDialect)

    module = Parser(ctx, source).parse_module()
    func_op = module.body.block.first_op
    assert isinstance(func_op, FuncOp)
    match configuration:
        case Configuration.SMT:
            semantics_op = func_op
        case Configuration.ARITH:
            semantics_module = module.clone()
            load_vanilla_semantics_using_control_flow_dialects()
            LowerToSMTPass().apply(ctx, semantics_module)
            LowerEffectPass().apply(ctx, semantics_module)
            SMTExpand().apply(ctx, semantics_module)
            semantics_op = semantics_module.body.block.first_op
            assert isinstance(semantics_op, FuncOp)

    return Pattern(func_op, semantics_op)


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
    return is_range_subset(left, merged_right)


def is_range_subset(left: SymProgram | FuncOp, right: SymProgram | FuncOp) -> bool:
    if isinstance(left, SymProgram) and isinstance(right, SymProgram):
        if not (left.fingerprint.may_be_subset(right.fingerprint)):
            return False
    if isinstance(left, SymProgram):
        left = left.func
    if isinstance(right, SymProgram):
        right = right.func

    match is_range_subset_with_z3(left, right, 2_000):
        case z3.sat:
            return True
        case z3.unsat:
            return False
        case _:
            pass

    num_cases = 1
    for op in left.walk():
        if isinstance(op, synth.ConstantOp):
            if op.res.type == smt.BoolType():
                num_cases *= 2
            elif isinstance(op.res.type, bv.BitVectorType):
                num_cases *= 1 << op.res.type.width.data

    if num_cases > 4096:
        return is_range_subset_with_z3(left, right) == z3.sat

    for case in range(num_cases):
        lhs_func = left.clone()
        for op in tuple(lhs_func.walk()):
            if isinstance(op, synth.ConstantOp):
                if op.res.type == smt.BoolType():
                    value = case % 2
                    case >>= 1
                    Rewriter.replace_op(op, smt.ConstantBoolOp(value == 1))
                elif isinstance(op.res.type, bv.BitVectorType):
                    value = case % (1 << op.res.type.width.data)
                    case >>= op.res.type.width.data
                    Rewriter.replace_op(op, bv.ConstantOp(value, op.res.type.width))
                else:
                    raise ValueError(f"Unsupported type: {op.res.type}")
        match is_range_subset_with_z3(lhs_func, right, 4_000):
            case z3.sat:
                continue
            case z3.unsat:
                return False
            case _:
                print("Failed with 4s timeout")
                break
    else:
        print("Succeeded with 4s timeouts")
        return True
    res = is_range_subset_with_z3(left, right)
    if res == z3.unknown:
        print("Failed with 25s timeout")
    return res == z3.sat


def is_range_subset_with_z3(
    left: FuncOp,
    right: FuncOp,
    timeout: int = 25_000,
) -> Any:
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
        forall_cst = builder.insert(
            smt.ForallOp(Region(Block(arg_types=lhs_cst_types)))
        )
        lhs_cst_args = forall_cst.body.block.args
        builder.insertion_point = InsertPoint.at_end(forall_cst.body.block)
        toplevel_val = forall_cst.result
    else:
        lhs_cst_args = ()

    # Create the rhs constant exists.
    rhs_cst_types = func_right.func_type.inputs.data[
        len(right.function_type.inputs.data) :
    ]
    if rhs_cst_types:
        exists_cst = builder.insert(
            smt.ExistsOp(Region(Block(arg_types=rhs_cst_types)))
        )
        rhs_cst_args = exists_cst.body.block.args
        if toplevel_val is not None:
            builder.insert(smt.YieldOp(exists_cst.result))
        else:
            toplevel_val = exists_cst.result
        builder.insertion_point = InsertPoint.at_end(exists_cst.body.block)
    else:
        rhs_cst_args = ()

    # Create the variable forall and the assert.
    if left.function_type.inputs.data:
        forall = builder.insert(
            smt.ForallOp(Region(Block(arg_types=left.function_type.inputs.data)))
        )
        var_args = forall.body.block.args
        if toplevel_val is not None:
            builder.insert(smt.YieldOp(forall.result))
        else:
            toplevel_val = forall.result
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

    return run_module_through_smtlib(
        module, timeout
    )  # pyright: ignore[reportUnknownVariableType]


@dataclass(frozen=True)
class SymFingerprint:
    fingerprint: dict[tuple[int, ...], dict[int, bool]]
    """
    For each list of input values, which values can be reached
    by the program by assigning constants to `xdsl.smt.synth`.
    """

    @staticmethod
    def _can_reach_result(
        func: FuncOp, inputs: tuple[int, ...], result: int
    ) -> bool | None:
        """
        Returns whether the program can reach the given result with the given
        inputs.
        This is done by checking the formula `exists csts, func(inputs, csts) == result`.
        """
        module = ModuleOp([])
        builder = Builder(InsertPoint.at_end(module.body.block))

        # Clone both functions into a new module.
        smt_func = builder.insert(clone_func_to_smt_func_with_constants(func))
        func_input_types = smt_func.func_type.inputs.data
        func_inputs: list[SSAValue] = []
        for input, type in zip(inputs, func_input_types[: len(inputs)], strict=True):
            if isinstance(type, bv.BitVectorType):
                cst_op = builder.insert(bv.ConstantOp(input, type.width))
                func_inputs.append(cst_op.res)
                continue
            if isinstance(type, smt.BoolType):
                cst_op = builder.insert(smt.ConstantBoolOp(input != 0))
                func_inputs.append(cst_op.result)
                continue
            raise ValueError(f"Unsupported type: {type}")
        for type in func_input_types[len(inputs) :]:
            declare_cst_op = builder.insert(smt.DeclareConstOp(type))
            func_inputs.append(declare_cst_op.res)
        assert len(func_inputs) == len(smt_func.func_type.inputs)
        call = builder.insert(smt.CallOp(smt_func.ret, func_inputs)).res

        result_val: SSAValue
        output_type = func.function_type.outputs.data[0]
        if isinstance(output_type, bv.BitVectorType):
            result_val = builder.insert(bv.ConstantOp(result, output_type.width)).res
        elif isinstance(output_type, smt.BoolType):
            result_val = builder.insert(smt.ConstantBoolOp(result != 0)).result
        else:
            raise ValueError(f"Unsupported output type: {output_type}")

        check = builder.insert(smt.EqOp(call[0], result_val)).res
        builder.insert(smt.AssertOp(check))

        # Now that we have the module, run it through the Z3 solver.
        res = run_module_through_smtlib(module)
        if res == z3.sat:
            return True
        if res == z3.unsat:
            return False
        return None

    @staticmethod
    def compute_exact_from_func(func: FuncOp) -> SymFingerprint:
        # Possible inputs per argument.
        possible_inputs: list[list[int]] = []
        for type in [*func.function_type.inputs, func.function_type.outputs.data[0]]:
            if isinstance(type, smt.BoolType):
                possible_inputs.append([0, -1])
                continue
            if isinstance(type, bv.BitVectorType):
                width = type.width.data
                possible_inputs.append([])
                for value in range(1 << width):
                    possible_inputs[-1].append(value)
                continue
            raise ValueError(f"Unsupported type: {type}")

        fingerprint: dict[tuple[int, ...], dict[int, bool]] = {}
        for input_values in itertools.product(*possible_inputs):
            res = SymFingerprint._can_reach_result(
                func, input_values[:-1], input_values[-1]
            )
            if res is not None:
                fingerprint.setdefault(input_values[:-1], {})[input_values[-1]] = res

        return SymFingerprint(fingerprint)

    @staticmethod
    def compute_from_func(func: FuncOp) -> SymFingerprint:
        num_possibilities = 1
        for type in [*func.function_type.inputs, func.function_type.outputs.data[0]]:
            if isinstance(type, smt.BoolType):
                num_possibilities *= 2
                continue
            if isinstance(type, bv.BitVectorType):
                width = type.width.data
                num_possibilities *= 1 << width
                continue
            raise ValueError(f"Unsupported type: {type}")
        if num_possibilities <= 16:
            return SymFingerprint.compute_exact_from_func(func)

        # Possible inputs per argument.
        possible_inputs: list[list[int]] = []
        for type in [*func.function_type.inputs, func.function_type.outputs.data[0]]:
            if isinstance(type, smt.BoolType):
                possible_inputs.append([0, -1])
                continue
            if isinstance(type, bv.BitVectorType):
                width = type.width.data
                possible_inputs.append(
                    list(
                        sorted(
                            {
                                0,
                                # 1,
                                # 2,
                                # (1 << width) - 1,
                                # 1 << (width - 1),
                                # (1 << (width - 1)) - 1,
                            }
                        )
                    )
                )
                continue
            raise ValueError(f"Unsupported type: {type}")

        fingerprint: dict[tuple[int, ...], dict[int, bool]] = {}
        for input_values in itertools.product(*possible_inputs):
            res = SymFingerprint._can_reach_result(
                func, input_values[:-1], input_values[-1]
            )
            if res is not None:
                fingerprint.setdefault(input_values[:-1], {})[input_values[-1]] = res

        return SymFingerprint(fingerprint)

    def short_string(self) -> str:
        """
        Returns a short string used to quickly compare two fingerprints.
        """
        return (
            "{{"
            + ",".join(
                f"[{','.join(map(str, inputs))}]:{{{','.join(str(result) if value else '!' + str(result) for result, value in results.items())}}}"
                for inputs, results in sorted(self.fingerprint.items())
            )
            + "}}"
        )

    def may_be_subset(self, other: SymFingerprint) -> bool:
        """
        Returns whether this fingerprint can represent a program that has a smaller
        range than the program represented by the other fingerprint.
        """
        for (lhs_inputs, lhs_results), (rhs_inputs, rhs_results) in zip(
            self.fingerprint.items(), other.fingerprint.items()
        ):
            if lhs_inputs != rhs_inputs:
                return False
            for result, value in lhs_results.items():
                if value:
                    if not rhs_results.get(result, True):
                        return False
        return True


def find_new_behaviors_in_bucket(
    canonicals: dict[UnorderedFingerprint, list[Pattern]],
    bucket: list[Pattern],
) -> tuple[dict[Pattern, list[Pattern]], list[list[Pattern]]]:
    # Sort programs into actual behavior buckets.
    behaviors: list[list[Pattern]] = []
    for pattern in bucket:
        for behavior in behaviors:
            if pattern.is_same_behavior(behavior[0]):
                behavior.append(pattern)
                break
        else:
            behaviors.append([pattern])

    # Exclude known behaviors.
    known_behaviors: dict[Pattern, list[Pattern]] = {}
    new_behaviors: list[list[Pattern]] = []
    for behavior in behaviors:
        for canonical in canonicals.get(behavior[0].unordered_fingerprint, []):
            if behavior[0].is_same_behavior(canonical):
                known_behaviors[canonical] = behavior
                break
        else:
            new_behaviors.append(behavior)
    return known_behaviors, new_behaviors


def find_new_behaviors(
    buckets: list[list[Pattern]],
    canonicals: list[Pattern],
) -> tuple[dict[Pattern, list[Pattern]], list[list[Pattern]]]:
    """
    Returns a `known_behaviors, new_behaviors` pair where `known_behaviors` is a
    map from canonical programs to buckets of new programs with the same
    behavior, and `new_behaviors` is a list of equivalence classes of the
    programs exhibiting a new behavior.
    """

    canonicals_dict: dict[UnorderedFingerprint, list[Pattern]] = {}
    for canonical in canonicals:
        canonicals_dict.setdefault(canonical.unordered_fingerprint, []).append(
            canonical
        )

    known_behaviors: dict[Pattern, list[Pattern]] = dict[Pattern, list[Pattern]]()
    new_behaviors: list[list[Pattern]] = []

    with Pool() as p:
        for i, (known, new) in enumerate(
            p.imap(partial(find_new_behaviors_in_bucket, canonicals_dict), buckets)
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
    new_canonicals: list[Pattern], new_rewrites: dict[Pattern, list[Pattern]]
) -> tuple[dict[Pattern, list[Pattern]], int]:
    buffer = StringIO()
    print("module {", file=buffer)
    print("module {", file=buffer)
    for canonical in new_canonicals:
        print("module{", file=buffer)
        print(canonical.func, file=buffer)
        print("}", file=buffer)
    print("}", file=buffer)
    print("module {", file=buffer)
    for programs in new_rewrites.values():
        for program in programs:
            print("module{", file=buffer)
            print(program.func, file=buffer)
            print("}", file=buffer)
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

    pruned_rewrites: dict[Pattern, list[Pattern]] = {
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
    def from_buckets(cls, phase: int, buckets: Iterable[list[Pattern]]):
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


@dataclass
class SymProgram:
    func: FuncOp
    fingerprint: SymFingerprint

    def __init__(self, func: FuncOp):
        self.func = func
        self.fingerprint = SymFingerprint.compute_from_func(func)


def parse_sym_program(source: str) -> SymProgram:
    ctx = Context()
    ctx.allow_unregistered = True
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(SMTDialect)
    ctx.load_dialect(SMTBitVectorDialect)
    ctx.load_dialect(SMTUtilsDialect)
    ctx.load_dialect(synth.SynthDialect)
    module = Parser(ctx, source).parse_module(True)
    func = module.body.block.first_op
    assert isinstance(func, FuncOp)
    func.detach()

    return SymProgram(func)


def main() -> None:
    global_start = time.time()

    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    canonicals: list[Pattern] = []
    illegals: list[Pattern] = []
    rewrites: list[RewriteRule] = []
    bucket_stats: list[BucketStat] = []

    try:
        for phase in range(args.phases + 1):
            phase_start = time.time()

            print(f"\033[1m== Phase {phase} (size at most {phase}) ==\033[0m")

            enumerating_start = time.time()
            buckets: dict[UnorderedFingerprint, list[Pattern]] = {}
            enumerated_count = 0
            building_blocks: list[list[FuncOp]] = []
            if phase >= 2:
                size = canonicals[0].size
                building_blocks.append([])
                for program in canonicals:
                    if program.size != size:
                        building_blocks.append([])
                        size = program.size
                    building_blocks[-1].append(program.func)
            illegal_patterns = list[pdl.PatternOp]()
            for illegal in illegals:
                body, _, root, _ = func_to_pdl(illegal.func)
                body.block.add_op(pdl.RewriteOp(root))
                pattern = pdl.PatternOp(1, None, body)
                illegal_patterns.append(pattern)

            with Pool() as p:
                for pattern in p.imap(
                    partial(parse_program, args.configuration),
                    enumerate_programs(
                        args.max_num_args,
                        phase,
                        args.bitvector_widths,
                        building_blocks if phase >= 2 else None,
                        illegal_patterns,
                        args.dialect,
                        args.configuration.value,
                    ),
                ):
                    if args.enumeration_order.phase(pattern) != phase:
                        continue
                    enumerated_count += 1
                    print(
                        f"\033[2K Enumerating programs... ({enumerated_count})",
                        end="\r",
                    )
                    fingerprint = pattern.unordered_fingerprint
                    if fingerprint not in buckets:
                        buckets[fingerprint] = []
                    buckets[fingerprint].append(pattern)
            enumerating_time = round(time.time() - enumerating_start, 2)
            print(
                f"\033[2KGenerated {enumerated_count} programs of this size "
                f"in {enumerating_time:.02f} s."
            )
            bucket_stats.append(BucketStat.from_buckets(phase, buckets.values()))

            new_rewrites: dict[Pattern, list[Pattern]] = {}

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
            new_canonicals: list[Pattern] = []
            for i, behavior in enumerate(new_behaviors):
                print(
                    f"\033[2K Choosing new canonical programs... "
                    f"({i + 1}/{len(new_behaviors)})",
                    end="\r",
                )
                canonical = min(
                    behavior, key=lambda p: next(iter(p.ordered_patterns()))
                )
                new_canonicals.append(canonical)
                new_rewrites[canonical] = behavior
            canonicals.extend(new_canonicals)
            # Sort canonicals to ensure deterministic output.
            canonicals.sort(key=lambda p: next(iter(p.ordered_patterns())))
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
                    f.write(str(program.func))
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

            cst_canonicals = list[SymProgram]()
            cst_illegals = list[FuncOp]()
            for phase in range(args.phases + 1):
                programs = list[SymProgram]()
                print("Enumerating programs in phase", phase)

                illegal_patterns = list[pdl.PatternOp]()
                for illegal in [illegal.func() for illegal in illegals] + cst_illegals:
                    body, _, root, _ = func_to_pdl(illegal)
                    body.block.add_op(pdl.RewriteOp(root))
                    pattern = pdl.PatternOp(1, None, body)
                    illegal_patterns.append(pattern)

                with Pool() as p:
                    for program in p.imap(
                        parse_sym_program,
                        enumerate_programs(
                            args.max_num_args,
                            phase,
                            args.bitvector_widths,
                            None,
                            illegal_patterns,
                            args.dialect,
                            ["--constant-kind=synth"],
                        ),
                    ):
                        should_skip = False
                        for canonical in cst_canonicals:
                            if program.func.is_structurally_equivalent(canonical.func):
                                should_skip = True
                                break
                        for illegal in cst_illegals:
                            if program.func.is_structurally_equivalent(illegal):
                                should_skip = True
                                break
                        if not should_skip:
                            programs.append(program)
                        print("Enumerated", len(programs), "programs", end="\r")
                print()

                # Group canonical programs by their function type, and merge them using
                # synth.constant.
                grouped_cst_canonicals: dict[FunctionType, list[SymProgram]] = {}
                for canonical in cst_canonicals:
                    grouped_cst_canonicals.setdefault(
                        canonical.func.function_type, []
                    ).append(canonical)

                new_illegals = 0
                new_possible_canonicals = list[SymProgram]()
                for program_idx, program in enumerate(programs):
                    canonicals_with_same_type = grouped_cst_canonicals.get(
                        program.func.function_type, []
                    )
                    if not canonicals_with_same_type:
                        new_possible_canonicals.append(program)
                        continue
                    if False:
                        for canonical_idx, canonical in enumerate(
                            canonicals_with_same_type
                        ):
                            print(
                                f"\033[2K Checking program {program_idx + 1}/{len(programs)} against old programs {canonical_idx + 1}/{len(canonicals_with_same_type)}",
                                end="\r",
                            )
                            assert (
                                program.func.function_type
                                == canonical.func.function_type
                            )

                            if is_range_subset(program, canonical):
                                print("Found illegal pattern:", end="")
                                pretty_print_func(program.func)
                                print("which is a subset of:", end="")
                                pretty_print_func(canonical.func)
                                print("")
                                cst_illegals.append(program.func)
                                break
                        else:
                            new_possible_canonicals.append(program)
                    else:
                        if is_range_subset_of_list_with_z3(
                            program.func, [p.func for p in canonicals_with_same_type]
                        ):
                            print("Found illegal pattern:", end="")
                            pretty_print_func(program.func)
                            new_illegals += 1
                            print("")
                            print(
                                f"Total illegal patterns found so far: {new_illegals} / {program_idx}"
                            )
                            cst_illegals.append(program.func)
                        else:
                            new_possible_canonicals.append(program)

                print()

                is_illegal_mask: list[bool] = [False] * len(new_possible_canonicals)
                for lhs_idx, lhs in enumerate(new_possible_canonicals):
                    if False:
                        for rhs_idx, rhs in enumerate(new_possible_canonicals):
                            print(
                                f"\033[2K Checking program for canonical {lhs_idx + 1}/{len(new_possible_canonicals)} against {rhs_idx + 1}/{len(new_possible_canonicals)}",
                                end="\r",
                            )

                            if is_illegal_mask[rhs_idx]:
                                continue
                            if lhs.func.function_type != rhs.func.function_type:
                                continue
                            if lhs is rhs:
                                continue
                            if is_range_subset(lhs, rhs):
                                cst_illegals.append(lhs.func)
                                is_illegal_mask[lhs_idx] = True
                                print("Found illegal pattern:", end="")
                                pretty_print_func(lhs.func)
                                print("which is a subset of:", end="")
                                pretty_print_func(rhs.func)
                                print("")
                                print(
                                    len([mask for mask in is_illegal_mask if mask]),
                                    "illegal patterns found so far",
                                )
                                print("")
                                break
                        else:
                            cst_canonicals.append(lhs)
                    else:
                        print(
                            f"\033[2K Checking program for canonical {lhs_idx + 1}/{len(new_possible_canonicals)}",
                            end="\r",
                        )
                        candidates: list[SymProgram] = []
                        for rhs_idx, rhs in enumerate(new_possible_canonicals):
                            if lhs_idx == rhs_idx:
                                continue
                            if is_illegal_mask[rhs_idx]:
                                continue
                            if lhs.func.function_type != rhs.func.function_type:
                                continue
                            candidates.append(rhs)
                        if candidates and is_range_subset_of_list_with_z3(
                            lhs.func, [c.func for c in candidates]
                        ):
                            cst_illegals.append(lhs.func)
                            is_illegal_mask[lhs_idx] = True
                            print("Found illegal pattern:", end="")
                            pretty_print_func(lhs.func)
                            print("")
                            print(
                                len([mask for mask in is_illegal_mask if mask]),
                                "illegal patterns found so far",
                            )
                            print("")
                        else:
                            cst_canonicals.append(lhs)
                print(f"== At step {phase} ==")
                print("number of canonicals", len(cst_canonicals))
                for canonical in cst_canonicals:
                    print("  ", end="")
                    pretty_print_func(canonical.func)
                    print("")
                print("number of illegals", len(cst_illegals))

                illegal_patterns = list[pdl.PatternOp]()
                for illegal in [illegal.func() for illegal in illegals] + cst_illegals:
                    body, _, root, _ = func_to_pdl(illegal)
                    body.block.add_op(pdl.RewriteOp(root))
                    pattern = pdl.PatternOp(1, None, body)
                    illegal_patterns.append(pattern)
                print(EXCLUDE_SUBPATTERNS_FILE)
                with open(EXCLUDE_SUBPATTERNS_FILE, "w") as f:
                    for illegal in illegal_patterns:
                        f.write(str(illegal))
                        f.write("\n// -----\n")

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
