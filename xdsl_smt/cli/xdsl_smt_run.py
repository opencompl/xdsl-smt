import argparse
import sys
from collections.abc import Sequence
from typing import Any, Literal

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.ir import Attribute
from xdsl.parser import Parser
from xdsl.tools.command_line_tool import CommandLineTool
from xdsl.traits import CallableOpInterface

from xdsl.interpreters.func import FuncFunctions
from xdsl_smt.interpreters.smt import SMTFunctions
from xdsl_smt.interpreters.smt_bitvector import SMTBitVectorFunctions
from xdsl_smt.interpreters.smt_utils import SMTUtilsFunctions

from xdsl_smt.dialects import (
    smt_dialect as smt,
    smt_bitvector_dialect as smt_bv,
    smt_utils_dialect as smt_utils,
    smt_array_dialect as smt_array,
)


def build_interpreter(module: ModuleOp, index_bitwidth: Literal[32, 64]) -> Interpreter:
    module.verify()

    interpreter = Interpreter(module, index_bitwidth=index_bitwidth)
    interpreter.register_implementations(FuncFunctions())
    interpreter.register_implementations(SMTFunctions())
    interpreter.register_implementations(SMTBitVectorFunctions())
    interpreter.register_implementations(SMTUtilsFunctions())

    return interpreter


def arity(interpreter: Interpreter) -> int:
    op = interpreter.get_op_for_symbol("main")
    trait = op.get_trait(CallableOpInterface)
    assert trait is not None
    return len(trait.get_argument_types(op))


def interpret(
    interpreter: Interpreter, arguments: Sequence[Attribute]
) -> tuple[Any, ...]:
    op = interpreter.get_op_for_symbol("main")
    trait = op.get_trait(CallableOpInterface)
    assert trait is not None
    argument_types = trait.get_argument_types(op)

    if len(arguments) != len(argument_types):
        raise ValueError(
            f"Wrong number of arguments provided (expected {len(argument_types)}, found {len(arguments)})"
        )

    args = tuple(
        interpreter.value_for_attribute(attr, attr_type)
        for attr, attr_type in zip(arguments, argument_types)
    )
    return interpreter.call_op(op, args)


class xDSLRunMain(CommandLineTool):
    def __init__(
        self,
        description: str = "xDSL-SMT runner",
        args: Sequence[str] | None = None,
    ):
        self.available_frontends = {}

        self.ctx = Context()
        self.register_all_dialects()
        self.register_all_frontends()
        # arg handling
        arg_parser = argparse.ArgumentParser(description=description)
        self.register_all_arguments(arg_parser)
        self.args = arg_parser.parse_args(args=args)

        self.ctx.allow_unregistered = self.args.allow_unregistered_dialect

    def register_all_dialects(self):
        super().register_all_dialects()
        del self.ctx._registered_dialects["smt"]  # pyright: ignore[reportPrivateUsage]
        self.ctx.register_dialect(smt.SMTDialect.name, lambda: smt.SMTDialect)
        self.ctx.register_dialect(
            smt_bv.SMTBitVectorDialect.name, lambda: smt_bv.SMTBitVectorDialect
        )
        self.ctx.register_dialect(smt_array.SMTArray.name, lambda: smt_array.SMTArray)
        self.ctx.register_dialect(
            smt_utils.SMTUtilsDialect.name, lambda: smt_utils.SMTUtilsDialect
        )
        self.ctx.load_registered_dialect(smt_utils.SMTUtilsDialect.name)
        self.ctx.load_registered_dialect(smt_bv.SMTBitVectorDialect.name)

    def register_all_arguments(self, arg_parser: argparse.ArgumentParser):
        arg_parser.add_argument(
            "--index-bitwidth",
            choices=(32, 64),
            default=Interpreter.DEFAULT_BITWIDTH,
            type=int,
            nargs="?",
            help="Bitwidth of the index type representation.",
        )
        arg_parser.add_argument(
            "--args",
            default="",
            type=str,
            help="Arguments to pass to entry function. Comma-separated list of xDSL "
            "Attributes, that will be parsed and converted by the interpreter.",
        )
        return super().register_all_arguments(arg_parser)

    def run(self):
        input, file_extension = self.get_input_stream()
        try:
            module = self.parse_chunk(input, file_extension)
            if module is None:
                return
            arg_parser = Parser(self.ctx, self.args.args, "args")
            runner_args = arg_parser.parse_optional_undelimited_comma_separated_list(
                arg_parser.parse_optional_attribute, arg_parser.parse_attribute
            )
            if runner_args is None:
                runner_args = ()
            interpreter = build_interpreter(module, self.args.index_bitwidth)
            result = interpret(interpreter, runner_args)
            if result:
                if len(result) == 1:
                    print(f"{result[0]}")
                else:
                    print("(")
                    print(",\n".join(f"    {res}" for res in result))
                    print(")")
            else:
                print("()")
        finally:
            if input is not sys.stdin:
                input.close()


def main():
    return xDSLRunMain().run()


if __name__ == "__main__":
    main()
