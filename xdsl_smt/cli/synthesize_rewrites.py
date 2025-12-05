#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess as sp
import sys
import time
from dataclasses import dataclass, fields
from enum import Enum, auto
from functools import partial
from io import StringIO
from multiprocessing import Pool
from typing import (
    Any,
    Iterable,
    TypeVar,
)

from xdsl.context import Context
from xdsl.ir.core import Operation, SSAValue
from xdsl.parser import Parser
from xdsl.printer import Printer

from xdsl_smt.superoptimization.pattern import (
    Pattern,
    UnorderedFingerprint,
    OrderedPattern,
)
from xdsl_smt.utils.pdl import func_to_pdl
from xdsl_smt.superoptimization.program_enumeration import enumerate_programs

from xdsl_smt.dialects import get_all_dialects
from xdsl_smt.dialects import smt_bitvector_dialect as bv
from xdsl.dialects import pdl
from xdsl.dialects.builtin import (
    ModuleOp,
)
from xdsl.dialects.func import FuncOp, ReturnOp

from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import (
    load_vanilla_semantics_using_control_flow_dialects,
)
from xdsl_smt.passes.smt_expand import SMTExpand
from xdsl_smt.passes.lower_to_smt.lower_to_smt import LowerToSMTPass
from xdsl_smt.passes.lower_effects import LowerEffectPass
from xdsl_smt.passes.lower_pairs import LowerPairs

NUM_PROCESSES: int | None = None

sys.setrecursionlimit(100000)

MLIR_ENUMERATE = "./mlir-fuzz/build/bin/mlir-enumerate"
REMOVE_REDUNDANT_PATTERNS = "./mlir-fuzz/build/bin/remove-redundant-patterns"
SMT_MLIR = "./mlir-fuzz/dialects/smt.mlir"
EXCLUDE_SUBPATTERNS_FILE = f"/tmp/exclude-subpatterns-{time.time()}.mlir"
BUILDING_BLOCKS_FILE = f"/tmp/building-blocks-{time.time()}.mlir"


T = TypeVar("T")

Result = tuple[Any, ...]
Image = tuple[Result, ...]


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

        num_ops_before = len(
            ["op" for op in self._lhs.func.body.block.ops if "constant" not in op.name]
        )
        num_ops_after = len(
            ["op" for op in self._rhs.func.body.block.ops if "constant" not in op.name]
        )
        benefit = num_ops_before - num_ops_after

        lhs, args, left_root, _ = func_to_pdl(self._lhs.func)
        assert left_root is not None
        pattern = pdl.PatternOp(benefit, None, lhs)

        # Unify LHS and RHS arguments.
        arguments: list[SSAValue | None] = [None] * self._rhs.arity
        for (k, _), (k2, _) in zip(
            self._rhs.useful_parameters(), self._lhs.useful_parameters(), strict=True
        ):
            arguments[k] = args[k2]
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
        "--out",
        type=str,
        help="the directory in which to write the generated files",
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
        choices=tuple(c for c in Configuration),
        default=Configuration.SMT,
    )

    arg_parser.add_argument(
        "--consider-refinements",
        dest="consider_refinements",
        action="store_true",
        help="if present, check for refinements to reduce the number of canonical programs",
    )


def parse_program(configuration: Configuration, source: str) -> Pattern:
    ctx = Context()
    ctx.allow_unregistered = True
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

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
            LowerPairs().apply(ctx, semantics_module)
            semantics_op = semantics_module.body.block.first_op
            assert isinstance(semantics_op, FuncOp)

    return Pattern(func_op, semantics_op)


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

    with Pool(processes=NUM_PROCESSES) as p:
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
    new_canonicals: list[Pattern],
    new_rewrites: dict[Pattern, list[Pattern]],
    new_refinements: list[tuple[Pattern, Pattern]],
) -> tuple[dict[Pattern, list[Pattern]], list[tuple[Pattern, Pattern]], int]:
    buffer = StringIO()
    printer = Printer(buffer, print_generic_format=True)
    printer.print_string("module {")
    printer.print_string("module {")
    for canonical in new_canonicals:
        printer.print_string("module {")
        printer.print_op(canonical.func)
        printer.print_string("}")
    printer.print_string("}")
    printer.print_string("module {")
    for programs in new_rewrites.values():
        for program in programs:
            printer.print_string("module {")
            printer.print_op(program.func)
            printer.print_string("}")
    for _, program in new_refinements:
        printer.print_string("module {")
        printer.print_op(program.func)
        printer.print_string("}")
    printer.print_string("}")
    printer.print_string("}")
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
    pruned_refinements: list[tuple[Pattern, Pattern]] = []
    for program, refined in new_refinements:
        if res_lines[i] == "true":
            pruned_count += 1
        else:
            pruned_refinements.append((program, refined))
        i += 1
    return pruned_rewrites, pruned_refinements, pruned_count


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

            with Pool(processes=NUM_PROCESSES) as p:
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
                        additional_options=["--constant-kind=none"],
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
                behavior.remove(canonical)
                if behavior:
                    new_rewrites[canonical] = behavior
            choosing_time = round(time.time() - choosing_start, 2)
            print(
                f"\033[2KChose {len(new_canonicals)} new canonical programs "
                f"in {choosing_time:.02f} s."
            )

            new_refinements: list[tuple[Pattern, Pattern]] = []
            if args.consider_refinements and args.phases < 2:
                print("Checking for refinements between canonicals:")
                index: int = 0
                num_canonicals = len(new_canonicals)
                for i, pattern in enumerate(new_canonicals.copy()):
                    print(
                        f"\033Checking for refinements: [2K  ({i + 1}/{num_canonicals})",
                        end="\r",
                    )
                    for pattern2 in new_canonicals:
                        if pattern is pattern2:
                            continue
                        if pattern2.is_refinement(pattern):
                            new_refinements.append((pattern2, pattern))
                            del new_canonicals[index]
                            break
                    else:
                        index += 1

            canonicals.extend(new_canonicals)
            # Sort canonicals to ensure deterministic output.
            canonicals.sort(key=lambda p: next(iter(p.ordered_patterns())))

            print(" Removing redundant illegal sub-patterns...", end="\r")
            pruning_start = time.time()
            (
                pruned_rewrites,
                pruned_refinements,
                pruned_count,
            ) = remove_redundant_illegal_subpatterns(
                new_canonicals, new_rewrites, new_refinements
            )
            for new_illegals in pruned_rewrites.values():
                illegals.extend(new_illegals)
            for _, new_illegal in pruned_refinements:
                illegals.append(new_illegal)
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

        if args.out != "":
            os.makedirs(args.out, exist_ok=True)
            with open(
                os.path.join(args.out, "canonicals.mlir"), "w", encoding="UTF-8"
            ) as f:
                for program in canonicals:
                    f.write(str(program.func))
                    f.write("\n// -----\n")

            module = ModuleOp([rewrite.to_pdl() for rewrite in rewrites])
            with open(
                os.path.join(args.out, "rewrites.mlir"), "w", encoding="UTF-8"
            ) as f:
                f.write("module {")
                for rewrite in rewrites:
                    f.write("\n\n\n")
                    f.write("// Input program:\n")
                    for line in str(
                        rewrite._lhs.ordered_func  # pyright: ignore[reportPrivateUsage]
                    ).splitlines():
                        f.write(f"// {line}\n")
                    f.write("\n")
                    f.write("// Rewrite to:\n")
                    for line in str(
                        rewrite._rhs.ordered_func  # pyright: ignore[reportPrivateUsage]
                    ).splitlines():
                        f.write(f"// {line}\n")
                    f.write("\n")
                    f.write(str(rewrite.to_pdl()))
                    f.write("\n")
                f.write("}")

            module = ModuleOp([illegal.func.clone() for illegal in illegals])
            with open(
                os.path.join(args.out, "illegals.mlir"), "w", encoding="UTF-8"
            ) as f:
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
