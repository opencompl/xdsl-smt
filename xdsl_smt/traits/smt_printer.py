"""
Utilities to print SMT-LIB expressions written in xDSL to the
SMT-LIB format.
"""

from __future__ import annotations
from abc import abstractmethod
from collections import deque

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
    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx) -> None:
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

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx) -> None:
        assert isinstance(self, Operation)
        print(f"({self.op_name()}", file=stream, end="")
        for operand in self.operands:
            print(" ", file=stream, end="")
            ctx.print_expr_to_smtlib(operand, stream)
        print(")", file=stream, end="")

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
            base_name = f"${value}"
        elif isinstance(value, SSAValue) and value.name_hint is not None:
            base_name = f"${value.name_hint}"
        else:
            base_name = "$tmp"

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

    def _expr_operands_topo_sort(self, op: Operation) -> list[SSAValue]:
        """
        Get the order in which the let bindings should be introduced for the
        transitive uses of the value parent.
        """
        # The stack of values to add. The leftmost value is the value we are
        # currently considering to add
        stack: deque[SSAValue] = deque(op.operands)

        # The values we need to add in order, and the set of values that we have
        # already processed and that do not need to be added anymore
        let_values = list[SSAValue]()
        processed = set[SSAValue]()

        while stack:
            val = stack[0]

            # If the value is a variable, we don't need to add it as a let binding.
            if val in self.value_to_name.keys():
                stack.popleft()
                processed.add(val)
                continue

            # If the value is processed, we can continue.
            if val in processed:
                stack.popleft()
                continue

            # We get the operands of the value owner.
            assert isinstance(val.owner, Operation)
            operands_to_add = [
                operand for operand in val.owner.operands if operand not in processed
            ]

            # If there are still unprocessed operands, we should process them first.
            if operands_to_add:
                for operand in operands_to_add:
                    stack.appendleft(operand)
                continue

            # Process the value
            stack.popleft()

            # Otherwise, we add it to the let binding if they have more than one use
            if len(val.uses) > 1:
                let_values.append(val)
            processed.add(val)

        return let_values

    def print_expr_to_smtlib(
        self, val: SSAValue, stream: IO[str], identation: str = ""
    ) -> None:
        """
        Print the SSA value expression in the SMTLib format.
        """
        if val in self.value_to_name.keys():
            print(self.value_to_name[val], file=stream, end="")
            return

        # First, get all the values we are going to put in let bindings
        assert isinstance(val, OpResult)
        let_values = self._expr_operands_topo_sort(val.op)

        for idx, let_value in enumerate(let_values):
            assert isinstance(let_value.owner, SMTLibOp)
            name = self.get_fresh_name(let_value)
            if idx != 0:
                print(identation, file=stream, end="")
            print(f"(let (({name} ", file=stream, end="")
            let_value.owner.print_expr_to_smtlib(stream, self)
            print(")) ", file=stream)

        assert isinstance(val.op, SMTLibOp)
        if let_values:
            print(identation, file=stream, end="")
        val.op.print_expr_to_smtlib(stream, self)
        print(")" * len(let_values), file=stream, end="")
        for let_value in let_values:
            self.names.remove(self.value_to_name[let_value])
            del self.value_to_name[let_value]


def print_to_smtlib(module: ModuleOp, stream: IO[str]) -> None:
    """
    Print a program to its SMTLib representation.
    """
    ctx = SMTConversionCtx()
    # We use this hack for now
    # TODO: check for usage of pairs in the program to not always print this.
    print(
        "(declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))",
        file=stream,
    )
    for op in module.ops:
        if isinstance(op, SMTLibScriptOp):
            op.print_expr_to_smtlib(stream, ctx)
            continue
