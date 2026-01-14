import argparse
import subprocess as sp

from xdsl.printer import Printer
from xdsl.context import Context
from xdsl.ir import Operation
from xdsl.parser import Parser
from xdsl.dialects.builtin import ModuleOp

from xdsl_smt.superoptimization.program_enumeration import enumerate_programs
from xdsl_smt.dialects import get_all_dialects


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "--input-dialect", type=str, help="path to the input dialect", required=True
    )
    arg_parser.add_argument(
        "--input-configuration",
        type=str,
        help="path to the input configuration",
        required=True,
    )
    arg_parser.add_argument(
        "--output-dialect", type=str, help="path to the output dialect", required=True
    )
    arg_parser.add_argument(
        "--output-configuration",
        type=str,
        help="path to the output configuration",
        required=True,
    )
    arg_parser.add_argument(
        "--max-num-ops",
        type=int,
        help="maximum number of operations in the MLIR programs that are generated",
        required=True,
    )
    arg_parser.add_argument(
        "--timeout",
        type=int,
        help="The timeout passed to the SMT solver in milliseconds",
        default=8000,
    )
    arg_parser.add_argument(
        "--opt",
        help="Run some optimizations on the SMT query before passing it to the solver",
        action="store_true",
    )


def get_input_operations(
    arg_parser: argparse.Namespace, ctx: Context
) -> list[ModuleOp]:
    """
    Get a module for each op in the input dialect.
    Some ops may appear multiple times with different types and attributes.
    """
    op_list = list[ModuleOp]()
    for program in enumerate_programs(
        max_num_args=99,
        num_ops=1,
        bv_widths="8",
        building_blocks=None,
        illegals=[],
        dialect_path=arg_parser.input_dialect,
        configuration=arg_parser.input_configuration,
        additional_options=["--exact-size", "--constant-kind=none"],
    ):
        module = Parser(ctx, program).parse_module()

        # Only consider programs that do not reuse values.
        # This is because we do not care about `add(%x, %x)`, just `add(%x, %y)`.
        def should_add(op: Operation) -> bool:
            for op in module.walk():
                for operand in op.operands:
                    if operand.has_more_than_one_use():
                        return False
            return True

        if should_add(module):
            op_list.append(module)
    return op_list


def try_synthesize_lowering_for_module(
    module: ModuleOp, args: argparse.Namespace, size: int, ctx: Context
) -> ModuleOp | None:
    """
    Try to synthesize a lowering for the given module (containing a single function), given the maximum
    number of operations to output.
    """
    with open("/tmp/input-synthesize.mlir", "w") as f:
        Printer(f, print_generic_format=True).print_op(module)
    res = sp.run(
        [
            "superoptimize",
            "/tmp/input-synthesize.mlir",
            f"--max-num-ops={size}",
            f"--dialect={args.output_dialect}",
            f"--configuration={args.output_configuration}",
        ]
        + (["--opt"] if args.opt else []),
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return None
    return Parser(ctx, res.stdout).parse_module()


def main():
    ctx = Context()
    ctx.allow_unregistered = True
    # Register all dialects
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)

    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    op_list = []
    failed_synthesis: list[ModuleOp] = get_input_operations(args, ctx)

    for i in range(1, args.max_num_ops + 1):
        op_list = failed_synthesis
        failed_synthesis = []
        print(
            f"Trying to synthesize lowerings with up to {i} operations. {len(op_list)} remaining."
        )
        for op in op_list:
            synthesized = try_synthesize_lowering_for_module(op, args, i, ctx)
            if synthesized is None:
                failed_synthesis.append(op)
                print(f"Failed to synthesize lowering with {i} operations for:")
                print(op)
                print("\n\n")
                continue

            print("Successfully synthesized lowering for:")
            print(op)
            print("The synthesized lowering is:")
            print(synthesized)
            print("\n\n")
        print(f"{len(failed_synthesis)} remaining after size {i}.")

    print(len(failed_synthesis), "operations could not be lowered:")
    for op in failed_synthesis:
        print(op)
        print("\n\n")


if __name__ == "__main__":
    main()
