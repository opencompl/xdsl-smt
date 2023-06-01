from __future__ import annotations

from typing import Annotated, Sequence, TypeVar, IO

from xdsl.irdl import (
    OpAttr,
    OptOpAttr,
    SingleBlockRegion,
    VarOperand,
    irdl_attr_definition,
    irdl_op_definition,
    Operand,
    IRDLOperation,
)
from xdsl.ir import (
    Block,
    Dialect,
    OpResult,
    Data,
    Operation,
    ParametrizedAttribute,
    Attribute,
    SSAValue,
    Region,
    TypeAttribute,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.builtin import FunctionType, StringAttr
from xdsl.utils.exceptions import VerifyException

from traits.effects import Pure

from traits.smt_printer import (
    SMTLibOp,
    SMTLibScriptOp,
    SimpleSMTLibOp,
    SMTLibSort,
    SMTConversionCtx,
)


@irdl_attr_definition
class BoolType(ParametrizedAttribute, SMTLibSort, TypeAttribute):
    name = "smt.bool"

    def print_sort_to_smtlib(self, stream: IO[str]) -> None:
        print("Bool", file=stream, end="")


@irdl_op_definition
class YieldOp(IRDLOperation):
    """`smt.yield` is used to return a result from a region."""

    name = "smt.yield"
    ret: Annotated[Operand, BoolType]

    def __init__(self, ret: Operand | Operation):
        super().__init__(operands=[ret])


@irdl_op_definition
class ForallOp(IRDLOperation, Pure, SMTLibOp):
    """Universal quantifier."""

    name = "smt.forall"

    res: Annotated[OpResult, BoolType]
    body: SingleBlockRegion

    @staticmethod
    def from_variables(
        variables: Sequence[Attribute], body: Region | None = None
    ) -> ForallOp:
        if body is None:
            body = Region([Block(arg_types=variables)])
        return ForallOp.create(result_types=[BoolType()], regions=[body])

    def verify_(self) -> None:
        if len(self.body.ops) == 0 or not isinstance(self.body.block.last_op, YieldOp):
            raise VerifyException("Region does not end in yield")

    @property
    def return_val(self) -> SSAValue:
        yield_op = self.body.block.last_op
        if not isinstance(yield_op, YieldOp):
            raise ValueError("Region does not end in yield")
        return yield_op.ret

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        print("(forall (", file=stream, end="")
        for idx, param in enumerate(self.body.block.args):
            assert isinstance(param.typ, SMTLibSort)
            param_name = ctx.get_fresh_name(param)
            if idx != 0:
                print(" ", file=stream, end="")
            print(f"({param_name} ", file=stream, end="")
            param.typ.print_sort_to_smtlib(stream)
            print(")", file=stream, end="")
        print(") ", file=stream, end="")
        ctx.print_expr_to_smtlib(self.return_val, stream)
        print(")", file=stream, end="")


@irdl_op_definition
class ExistsOp(IRDLOperation, Pure, SMTLibOp):
    """Existential quantifier."""

    name = "smt.exists"

    res: Annotated[OpResult, BoolType]
    body: SingleBlockRegion

    @staticmethod
    def from_variables(
        variables: Sequence[Attribute], body: Region | None = None
    ) -> ExistsOp:
        if body is None:
            body = Region([Block(arg_types=variables)])
        return ExistsOp.create(result_types=[BoolType()], regions=[body])

    def verify_(self) -> None:
        if len(self.body.ops) == 0 or not isinstance(self.body.block.last_op, YieldOp):
            raise VerifyException("Region does not end in yield")

    @property
    def return_val(self) -> SSAValue:
        yield_op = self.body.block.last_op
        if not isinstance(yield_op, YieldOp):
            raise ValueError("Region does not end in yield")
        return yield_op.ret

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        print("(exists (", file=stream, end="")
        for idx, param in enumerate(self.body.blocks[0].args):
            assert isinstance(param.typ, SMTLibSort)
            param_name = ctx.get_fresh_name(param)
            if idx != 0:
                print(" ", file=stream, end="")
            print(f"({param_name} ", file=stream, end="")
            param.typ.print_sort_to_smtlib(stream)
            print(")", file=stream, end="")
        print(") ", file=stream, end="")
        ctx.print_expr_to_smtlib(self.return_val, stream)
        print(")", file=stream, end="")


@irdl_op_definition
class CallOp(IRDLOperation, Pure, SMTLibOp):
    """Call to an SMT function."""

    name = "smt.call"

    res: OpResult
    func: Annotated[Operand, FunctionType]
    args: VarOperand

    @staticmethod
    def get(func: Operand, args: list[Operand]) -> CallOp:
        if not isinstance(func.typ, FunctionType):
            raise Exception("Expected function type, got ", func.typ)
        return CallOp.build(
            operands=[func, args],
            result_types=[func.typ.outputs.data[0]],
        )

    def verify_(self) -> None:
        assert isinstance(self.func.typ, FunctionType)
        if len(self.args) != len(self.func.typ.inputs.data):
            raise VerifyException("Incorrect number of arguments")
        for arg, arg_type in zip(self.args, self.func.typ.inputs.data):
            if arg.typ != arg_type:
                raise VerifyException("Incorrect argument type")
        if len(self.func.typ.outputs.data) != 1:
            raise VerifyException("Incorrect number of return values")
        if self.res.typ != self.func.typ.outputs.data[0]:
            raise VerifyException("Incorrect return type")

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        print("(", file=stream, end="")
        for idx, operand in enumerate(self.operands):
            if idx != 0:
                print(" ", file=stream, end="")
            ctx.print_expr_to_smtlib(operand, stream)
        print(")", file=stream, end="")


################################################################################
#                             Script operations                                #
################################################################################


@irdl_op_definition
class DefineFunOp(IRDLOperation, SMTLibScriptOp):
    """Define a function."""

    name = "smt.define_fun"

    fun_name: OptOpAttr[StringAttr]
    ret: Annotated[OpResult, FunctionType]
    body: SingleBlockRegion

    def verify_(self) -> None:
        if len(self.body.ops) == 0 or not isinstance(self.body.block.last_op, ReturnOp):
            raise VerifyException("Region does not end in return")
        if len(self.body.blocks[0].args) != len(self.func_type.inputs.data):
            raise VerifyException("Incorrect number of arguments")
        for arg, arg_type in zip(self.body.blocks[0].args, self.func_type.inputs.data):
            if arg.typ != arg_type:
                raise VerifyException("Incorrect argument type")
        if len(self.func_type.outputs.data) != 1:
            raise VerifyException("Incorrect number of return values")
        if self.return_val.typ != self.func_type.outputs.data[0]:
            raise VerifyException("Incorrect return type")

    @property
    def func_type(self) -> FunctionType:
        """Get the function type of this operation."""
        if not isinstance(self.ret.typ, FunctionType):
            raise VerifyException("Incorrect return type")
        return self.ret.typ

    @property
    def return_val(self) -> SSAValue:
        """Get the return value of this operation."""
        ret_op = self.body.block.last_op
        if not isinstance(ret_op, ReturnOp):
            raise ValueError("Region does not end in a return")
        return ret_op.ret

    @staticmethod
    def from_function_type(
        func_type: FunctionType, name: str | StringAttr | None = None
    ):
        block = Block(arg_types=func_type.inputs.data)
        region = Region([block])
        if isinstance(name, str):
            name = StringAttr(name)
        if name is None:
            return DefineFunOp.create(result_types=[func_type], regions=[region])
        else:
            return DefineFunOp.build(
                result_types=[func_type], attributes={"name": name}, regions=[region]
            )

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        print("(define-fun ", file=stream, end="")

        # Print the function name
        name: str
        if self.fun_name is not None:
            name = ctx.get_fresh_name(self.fun_name.data)
            ctx.value_to_name[self.ret] = name
        else:
            name = ctx.get_fresh_name(self.ret)
        print(f"{name} ", file=stream, end="")

        # Print the function arguments
        print("(", file=stream, end="")
        for idx, arg in enumerate(self.body.blocks[0].args):
            if idx != 0:
                print(" ", file=stream, end="")
            arg_name = ctx.get_fresh_name(arg)
            typ = arg.typ
            assert isinstance(typ, SMTLibSort)
            print(f"({arg_name} ", file=stream, end="")
            typ.print_sort_to_smtlib(stream)
            print(")", file=stream, end="")
        print(") ", file=stream, end="")

        # Print the function return type
        assert len(self.func_type.outputs.data) == 1
        ret_type = self.func_type.outputs.data[0]
        assert isinstance(ret_type, SMTLibSort)
        ret_type.print_sort_to_smtlib(stream)
        print("", file=stream)

        # Print the function body
        print("  ", file=stream, end="")
        ctx.print_expr_to_smtlib(self.return_val, stream)
        print(")", file=stream)


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """The return operation of a function."""

    name = "smt.return"
    ret: Operand

    def __init__(self, operand: SSAValue):
        super().__init__(operands=[operand])

    @staticmethod
    def from_ret_value(ret_value: SSAValue):
        return ReturnOp.create(operands=[ret_value])

    def verify_(self):
        parent = self.parent_op()
        if not isinstance(parent, DefineFunOp):
            raise VerifyException("ReturnOp must be nested inside a DefineFunOp")
        if not self.ret.typ == parent.func_type.outputs.data[0]:
            raise VerifyException("ReturnOp type mismatch with DefineFunOp")


@irdl_op_definition
class DeclareConstOp(IRDLOperation, SMTLibScriptOp):
    """Declare a constant value."""

    name = "smt.declare_const"
    res: OpResult

    def __init__(self, type_name: Attribute):
        super().__init__(result_types=[type_name])

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        name = ctx.get_fresh_name(self.res)
        typ = self.res.typ
        assert isinstance(typ, SMTLibSort)
        print(f"(declare-const {name} ", file=stream, end="")
        typ.print_sort_to_smtlib(stream)
        print(")", file=stream)


@irdl_op_definition
class AssertOp(IRDLOperation, SMTLibScriptOp):
    """Assert that a boolean expression is true."""

    name = "smt.assert"
    op: Annotated[Operand, BoolType]

    def __init__(self, operand: SSAValue):
        super().__init__(operands=[operand])

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        print("(assert ", file=stream, end="")
        ctx.print_expr_to_smtlib(self.op, stream)
        print(")", file=stream)

    @staticmethod
    def get(arg: Operation | SSAValue) -> AssertOp:
        return AssertOp.build(operands=[arg])


@irdl_op_definition
class CheckSatOp(IRDLOperation, SMTLibScriptOp):
    """Check if the current set of assertions is satisfiable."""

    name = "smt.check_sat"

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        print("(check-sat)", file=stream)


# Core operations

_OpT = TypeVar("_OpT", bound=Operation)


class BinaryBoolOp(IRDLOperation, Pure):
    """Base class for binary boolean operations."""

    res: Annotated[OpResult, BoolType]
    lhs: Annotated[Operand, BoolType]
    rhs: Annotated[Operand, BoolType]

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[BoolType([])], operands=[lhs, rhs])

    @classmethod
    def get(cls: type[_OpT], lhs: SSAValue, rhs: SSAValue) -> _OpT:
        return cls.create(result_types=[BoolType([])], operands=[lhs, rhs])


class BinaryTOp(IRDLOperation, Pure):
    """Base class for binary operations with boolean results."""

    res: Annotated[OpResult, BoolType]
    lhs: Operand
    rhs: Operand

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[BoolType([])], operands=[lhs, rhs])

    @classmethod
    def get(cls: type[_OpT], lhs: SSAValue, rhs: SSAValue) -> _OpT:
        return cls.create(result_types=[BoolType([])], operands=[lhs, rhs])

    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ:
            raise ValueError("Operands must have the same type")


@irdl_attr_definition
class BoolAttr(Data[bool]):
    """Boolean value."""

    name = "smt.bool_attr"

    @staticmethod
    def parse_parameter(parser: Parser) -> bool:
        val = parser.expect(parser.try_parse_bare_id, "Expected 'true' or 'false'")
        if val.text == "true":
            return True
        if val.text == "false":
            return False
        raise ValueError("Expected 'true' or 'false'")

    def print_parameter(self, printer: Printer) -> None:
        printer.print("true" if self.data else "false")


@irdl_op_definition
class ConstantBoolOp(IRDLOperation, Pure, SMTLibOp):
    """Boolean constant."""

    name = "smt.constant_bool"

    res: Annotated[OpResult, BoolType]
    value: OpAttr[BoolAttr]

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        if self.value.data:
            print("true", file=stream, end="")
        else:
            print("false", file=stream, end="")

    @staticmethod
    def from_bool(value: bool) -> ConstantBoolOp:
        return ConstantBoolOp.create(
            result_types=[BoolType([])], attributes={"value": BoolAttr(value)}
        )


@irdl_op_definition
class NotOp(IRDLOperation, Pure, SimpleSMTLibOp):
    """Boolean negation."""

    name = "smt.not"

    res: Annotated[OpResult, BoolType]
    arg: Annotated[Operand, BoolType]

    @staticmethod
    def get(operand: SSAValue) -> NotOp:
        return NotOp.create(result_types=[BoolType()], operands=[operand])

    def op_name(self) -> str:
        return "not"


@irdl_op_definition
class ImpliesOp(BinaryBoolOp, SimpleSMTLibOp):
    """Boolean implication."""

    name = "smt.implies"

    def op_name(self) -> str:
        return "=>"


@irdl_op_definition
class AndOp(BinaryBoolOp, SimpleSMTLibOp):
    """Boolean conjunction."""

    name = "smt.and"

    def op_name(self) -> str:
        return "and"


@irdl_op_definition
class OrOp(BinaryBoolOp, SimpleSMTLibOp):
    """Boolean disjunction."""

    name = "smt.or"

    def op_name(self) -> str:
        return "or"


@irdl_op_definition
class XorOp(BinaryBoolOp, SimpleSMTLibOp):
    """Boolean exclusive disjunction."""

    name = "smt.xor"

    def op_name(self) -> str:
        return "xor"


@irdl_op_definition
class EqOp(BinaryTOp, SimpleSMTLibOp):
    """Equality."""

    name = "smt.eq"

    def op_name(self) -> str:
        return "="


@irdl_op_definition
class DistinctOp(BinaryTOp, SimpleSMTLibOp):
    """Distinctness."""

    name = "smt.distinct"

    def op_name(self) -> str:
        return "distinct"


@irdl_op_definition
class IteOp(IRDLOperation, Pure, SimpleSMTLibOp):
    """If-then-else."""

    name = "smt.ite"

    res: OpResult
    cond: Annotated[Operand, BoolType]
    true_val: Operand
    false_val: Operand

    def __init__(self, cond: SSAValue, true_val: SSAValue, false_val: SSAValue):
        super().__init__(
            result_types=[true_val.typ], operands=[cond, true_val, false_val]
        )

    def verify_(self) -> None:
        if not (self.res.typ == self.true_val.typ == self.false_val.typ):
            raise ValueError("The result and both values must have the same type")

    def op_name(self) -> str:
        return "ite"


SMTDialect = Dialect(
    [
        YieldOp,
        ForallOp,
        ExistsOp,
        CallOp,
        DefineFunOp,
        ReturnOp,
        DeclareConstOp,
        AssertOp,
        CheckSatOp,
        ConstantBoolOp,
        NotOp,
        ImpliesOp,
        AndOp,
        OrOp,
        XorOp,
        EqOp,
        DistinctOp,
        IteOp,
    ],
    [BoolType, BoolAttr],
)
