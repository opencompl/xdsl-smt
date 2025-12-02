from typing import Any
from io import StringIO
import sys

import z3  # pyright: ignore[reportMissingTypeStubs]

from xdsl.dialects.builtin import ModuleOp

from xdsl_smt.traits.smt_printer import print_to_smtlib


def run_module_through_smtlib(
    module: ModuleOp, timeout: int = 1000
) -> tuple[Any, z3.Solver]:
    smtlib_program = StringIO()
    print_to_smtlib(module, smtlib_program)

    # Parse the SMT-LIB program and run it through the Z3 solver.
    ctx = z3.Context()
    solver = z3.Solver(ctx=ctx)
    # Set the timeout
    solver.set("timeout", timeout)  # pyright: ignore[reportUnknownMemberType]
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
        raise e
    return result, solver
