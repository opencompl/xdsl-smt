from __future__ import annotations
from abc import abstractmethod

from dataclasses import dataclass, field
from typing import IO
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import OpResult, SSAValue, Operation


class SMTLibSort:
    """Mark an attribute to be an SMTLib sort."""

    @abstractmethod
    def print_sort_to_smtlib(self, stream: IO[str]) -> None:
        """Print an attribute to an SMTLib representation."""
        ...


class SMTLibOp:
    """
    Mark an operation to be an SMTLib expression.
    This include only expressions that have no side-effects,
    unlike script operations.
    """

    @abstractmethod
    def print_expr_to_smtlib(self, stream: IO[str],
                             ctx: SMTConversionCtx) -> None:
        """Print the operation to an SMTLib representation."""
        ...


class SMTLibScriptOp(SMTLibOp):
    """
    Mark an operation to be an SMTLib script operation.
    This include `check-sat`, or `assert` for instance.
    """
    pass


class SimpleSMTLibOp(SMTLibOp):
    """
    Marker of operations that can be printed as SMTLib expressions with format
    `(expr_name <arg0> <arg1> ... <argN>)`.
    """

    def print_expr_to_smtlib(self, stream: IO[str],
                             ctx: SMTConversionCtx) -> None:
        assert isinstance(self, Operation)
        print(f"({self.op_name()}", file=stream, end='')
        for operand in self.operands:
            print(" ", file=stream, end='')
            ctx.print_expr_to_smtlib(operand, stream)
        print(")", file=stream, end='')

    @abstractmethod
    def op_name(self) -> str:
        """Expression name when printed in SMTLib."""
        ...


@dataclass
class SMTConversionCtx:
    """
    Context keeping the names of variables when printed in SMTLib.
    This is used during the conversion from xDSL to SMTLib.
    """
    value_to_name: dict[SSAValue, str] = field(default_factory=dict)
    names: set[str] = field(default_factory=set)

    def get_fresh_name(self, value: str | SSAValue | None) -> str:
        """
        Get a fresh name given a base name. If an SSA value is given,
        the base name is taken from the SSAValue name.
        """
        base_name: str
        if isinstance(value, str):
            base_name = value
        elif isinstance(value, SSAValue) and value.name is not None:
            base_name = value.name
        else:
            base_name = "tmp"

        name: str
        if base_name not in self.names:
            name = base_name
        else:
            i = 0
            while f"{base_name}_{i}" in self.names:
                i += 1
            name = f"{base_name}_{i}"

        if isinstance(value, SSAValue):
            self.value_to_name[value] = name
        self.names.add(name)
        return name

    def print_expr_to_smtlib(self, val: SSAValue, stream: IO[str]) -> None:
        """
        Print the SSA value expression in the SMTLib format.
        """
        if val in self.value_to_name.keys():
            print(self.value_to_name[val], file=stream, end='')
            return
        assert isinstance(val, OpResult)
        op = val.op
        assert isinstance(op, SMTLibOp)
        op.print_expr_to_smtlib(stream, self)


def print_to_smtlib(module: ModuleOp, stream: IO[str]) -> None:
    """
    Print a program to its SMTLib representation.
    """
    ctx = SMTConversionCtx()
    # We use this hack for now
    # TODO: check for usage of pairs in the program to not always print this.
    print("(declare-datatypes ((Pair 2)) "
          "((par (X Y) ((pair (first X) (second Y))))))")
    for op in module.ops:
        if isinstance(op, SMTLibScriptOp):
            op.print_expr_to_smtlib(stream, ctx)
            continue
