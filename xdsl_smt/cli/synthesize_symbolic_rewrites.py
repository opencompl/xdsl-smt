from __future__ import annotations

import time
import argparse
import z3  # pyright: ignore[reportMissingTypeStubs]
import itertools
from enum import Enum
from typing import Any, Sequence
from dataclasses import dataclass
from multiprocessing import Pool

from xdsl.ir import SSAValue, Region, Block
from xdsl.parser import Parser
from xdsl.context import Context
from xdsl.builder import Builder
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects import pdl
from xdsl.dialects.builtin import FunctionType

from xdsl_smt.superoptimization.program_enumeration import enumerate_programs
from xdsl_smt.utils.pdl import func_to_pdl
from xdsl_smt.utils.inlining import inline_single_result_func
from xdsl_smt.utils.run_with_smt_solver import run_module_through_smtlib
from xdsl_smt.dialects import get_all_dialects
from xdsl_smt.dialects import (
    smt_dialect as smt,
    synth_dialect as synth,
    smt_bitvector_dialect as bv,
)

EXCLUDE_SUBPATTERNS_FILE = f"/tmp/exclude-subpatterns-{time.time()}.mlir"


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
        "--out-canonicals",
        type=str,
        help="the file in which to write the generated canonical programs",
        default="",
    )
    arg_parser.add_argument(
        "--out-illegals",
        type=str,
        help="the file in which to write the generated illegal programs",
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
        help="The IRDL file describing the dialect to use for enumeration",
    )

    arg_parser.add_argument(
        "--configuration",
        dest="configuration",
        type=Configuration,
        choices=tuple(Configuration),
        default=Configuration.SMT,
    )

    arg_parser.add_argument(
        "--illegals",
        dest="illegals",
        type=str,
        help="The file containing illegal patterns to exclude",
    )


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
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    module = Parser(ctx, source).parse_module(True)
    func = module.body.block.first_op
    assert isinstance(func, FuncOp)
    func.detach()

    return SymProgram(func)


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


def main():
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    ctx = Context()
    ctx.allow_unregistered = True
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

    with open(args.illegals, "r", encoding="UTF-8") as f:
        illegals_module = Parser(ctx, f.read()).parse_module()
    illegals = [op for op in illegals_module.body.block.ops if isinstance(op, FuncOp)]

    cst_canonicals = list[SymProgram]()
    cst_illegals = list[FuncOp]()
    for phase in range(args.phases + 1):
        programs = list[SymProgram]()
        print("Enumerating programs in phase", phase)

        illegal_patterns = list[pdl.PatternOp]()
        for illegal in [illegal for illegal in illegals] + cst_illegals:
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
                    args.configuration.value,
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
            grouped_cst_canonicals.setdefault(canonical.func.function_type, []).append(
                canonical
            )

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
                for canonical_idx, canonical in enumerate(canonicals_with_same_type):
                    print(
                        f"\033[2K Checking program {program_idx + 1}/{len(programs)} against old programs {canonical_idx + 1}/{len(canonicals_with_same_type)}",
                        end="\r",
                    )
                    assert program.func.function_type == canonical.func.function_type

                    if is_range_subset(program, canonical):
                        print("Found illegal pattern:", end="")
                        print(program.func)
                        print("which is a subset of:", end="")
                        print(canonical.func)
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
                    print(program.func)
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
                        print(lhs.func)
                        print("which is a subset of:", end="")
                        print(rhs.func)
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
                    print(lhs.func)
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
            print(canonical.func)
            print("")
        print("number of illegals", len(cst_illegals))

        illegal_patterns = list[pdl.PatternOp]()
        for illegal in [illegal for illegal in illegals] + cst_illegals:
            body, _, root, _ = func_to_pdl(illegal)
            body.block.add_op(pdl.RewriteOp(root))
            pattern = pdl.PatternOp(1, None, body)
            illegal_patterns.append(pattern)

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
