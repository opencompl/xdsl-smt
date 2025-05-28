import argparse
import sys
from collections.abc import Sequence

from xdsl.context import Context
from xdsl.interpreter import Interpreter
from xdsl.parser import Parser
from xdsl_smt.dialects import smt_dialect as smt
from xdsl.tools.command_line_tool import CommandLineTool
from xdsl_smt.interpreters.smt import SMTFunctions


class xDSLRunMain(CommandLineTool):
    interpreter: Interpreter

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
        self.ctx.register_dialect("smt", lambda: smt.SMTDialect)

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

    def register_implementations(self, interpreter: Interpreter):
        interpreter.register_implementations(SMTFunctions())

    def run(self):
        input, file_extension = self.get_input_stream()
        try:
            module = self.parse_chunk(input, file_extension)
            if module is None:
                return
            module.verify()
            interpreter = Interpreter(module, index_bitwidth=self.args.index_bitwidth)
            self.register_implementations(interpreter)
            function = module.ops.first
            if not isinstance(function, smt.DefineFunOp):
                raise ValueError(
                    "The module must contain a single DefineFunOp as the entry point."
                )
            parser = Parser(self.ctx, self.args.args, "args")
            runner_args = parser.parse_optional_undelimited_comma_separated_list(
                parser.parse_optional_attribute, parser.parse_attribute
            )
            if runner_args is None:
                runner_args = ()

            args = tuple(
                interpreter.value_for_attribute(attr, attr_type)
                for attr, attr_type in zip(runner_args, function.func_type.inputs)
            )
            result = interpreter.run_ssacfg_region(function.body, args, function.name)
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
