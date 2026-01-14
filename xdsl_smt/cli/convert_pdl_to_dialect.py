import argparse
from typing import Iterator

from xdsl.parser import Parser
from xdsl.context import Context
from xdsl.dialects import pdl

from xdsl_smt.dialects import get_all_dialects
from xdsl.dialects.builtin import StringAttr


def iterate_pdl_patterns(file_path: str, ctx: Context) -> Iterator[pdl.PatternOp]:
    with open(file_path, "r") as f:
        input_program = Parser(ctx, f.read()).parse_module()

    for pattern in input_program.walk():
        if isinstance(pattern, pdl.PatternOp):
            yield pattern


def convert_name_to_dialect(name: str) -> str | None:
    match name:
        case "smt.bv.add":
            return "arith.addi"
        case "smt.bv.sub":
            return "arith.subi"
        case "smt.bv.and":
            return "arith.andi"
        case "smt.bv.or":
            return "arith.ori"
        case "smt.bv.xor":
            return "arith.xori"
        case _:
            return None


def convert_pdl_to_dialect(pattern: pdl.PatternOp) -> pdl.PatternOp | None:
    new_pattern = pattern.clone()
    for op in new_pattern.walk():
        if isinstance(op, pdl.OperationOp):
            if op.opName is None:
                return None
            new_name = convert_name_to_dialect(op.opName.data)
            if new_name is None:
                return None
            op.opName = StringAttr(new_name)

    return new_pattern


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument("input_file", type=str, help="path to the input file")


def main():
    ctx = Context()
    ctx.allow_unregistered = True
    for dialect, factory in get_all_dialects().items():
        ctx.register_dialect(dialect, factory)

    parser = argparse.ArgumentParser(description="Convert PDL to dialect")
    register_all_arguments(parser)
    args = parser.parse_args()

    for pattern in iterate_pdl_patterns(args.input_file, ctx):
        if new_pattern := convert_pdl_to_dialect(pattern):
            print(pattern)
            print(new_pattern)


if __name__ == "__main__":
    main()
