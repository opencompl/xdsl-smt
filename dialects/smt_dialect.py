from __future__ import annotations

from io import IOBase
from typing import Annotated, TypeVar

from xdsl.irdl import (OpAttr, OptOpAttr, SingleBlockRegion, VarOperand,
                       irdl_attr_definition, irdl_op_definition, Operand)
from xdsl.ir import (Block, Dialect, OpResult, Operation, Data,
                     ParametrizedAttribute, Attribute, SSAValue, Region)
from xdsl.parser import BaseParser
from xdsl.printer import Printer
from xdsl.dialects.builtin import (FunctionType, StringAttr)
from xdsl.utils.exceptions import VerifyException

from .smt_printer_interface import (SMTLibOp, SMTLibScriptOp, SimpleSMTLibOp,
                                    SMTLibSort, SMTConversionCtx)


@irdl_attr_definition
class BoolType(ParametrizedAttribute, SMTLibSort):
    name = "smt.bool"

    def print_sort_to_smtlib(self, stream: IOBase) -> None:
        print("Bool", file=stream, end='')


@irdl_op_definition
class YieldOp(Operation):
    """`smt.yield` is used to return a result from a region."""

    name = "smt.yield"
    ret: Annotated[Operand, BoolType]

    @classmethod
    def parse(cls: type[YieldOp], result_types: list[Attribute],
              parser: BaseParser) -> YieldOp:
        ret = parser.parse_operand()
        return YieldOp.create(operands=[ret])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.ret)


@irdl_op_definition
class ForallOp(Operation, SMTLibOp):
    """Universal quantifier."""
    name = "smt.forall"

    res: Annotated[OpResult, BoolType]
    body: SingleBlockRegion

    def verify_(self) -> None:
        if (len(self.body.ops) == 0
                or not isinstance(self.body.ops[-1], YieldOp)):
            raise VerifyException("Region does not end in yield")

    @property
    def return_val(self) -> SSAValue:
        yield_op = self.body.ops[-1]
        if not isinstance(yield_op, YieldOp):
            raise ValueError("Region does not end in yield")
        return yield_op.ret

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(forall (", file=stream, end='')
        for idx, param in enumerate(self.body.blocks[0].args):
            assert isinstance(param.typ, SMTLibSort)
            param_name = ctx.get_fresh_name(param)
            if idx != 0:
                print(" ", file=stream, end='')
            print(f"({param_name} ", file=stream, end='')
            param.typ.print_sort_to_smtlib(stream)
            print(")", file=stream, end='')
        print(") ", file=stream, end='')
        ctx.print_expr_to_smtlib(self.return_val, stream)
        print(")", file=stream, end='')


@irdl_op_definition
class ExistsOp(Operation, SMTLibOp):
    """Existential quantifier."""
    name = "smt.exists"

    res: Annotated[OpResult, BoolType]
    body: SingleBlockRegion

    def verify_(self) -> None:
        if (len(self.body.ops) == 0
                or not isinstance(self.body.ops[-1], YieldOp)):
            raise VerifyException("Region does not end in yield")

    @property
    def return_val(self) -> SSAValue:
        yield_op = self.body.ops[-1]
        if not isinstance(yield_op, YieldOp):
            raise ValueError("Region does not end in yield")
        return yield_op.ret

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(exists (", file=stream, end='')
        for idx, param in enumerate(self.body.blocks[0].args):
            assert isinstance(param.typ, SMTLibSort)
            param_name = ctx.get_fresh_name(param)
            if idx != 0:
                print(" ", file=stream, end='')
            print(f"({param_name} ", file=stream, end='')
            param.typ.print_sort_to_smtlib(stream)
            print(")", file=stream, end='')
        print(") ", file=stream, end='')
        ctx.print_expr_to_smtlib(self.return_val, stream)
        print(")", file=stream, end='')


@irdl_op_definition
class CallOp(Operation, SMTLibOp):
    """Call to an SMT function."""
    name = "smt.call"

    res: OpResult
    func: Annotated[Operand, FunctionType]
    args: VarOperand

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

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(", file=stream, end='')
        for idx, operand in enumerate(self.operands):
            if idx != 0:
                print(" ", file=stream, end='')
            ctx.print_expr_to_smtlib(operand, stream)
        print(")", file=stream, end='')


################################################################################
#                             Script operations                                #
################################################################################


@irdl_op_definition
class DefineFunOp(Operation, SMTLibScriptOp):
    """Define a function."""
    name = "smt.define_fun"

    fun_name: OptOpAttr[StringAttr]
    ret: Annotated[OpResult, FunctionType]
    body: SingleBlockRegion

    def verify_(self) -> None:
        if (len(self.body.ops) == 0
                or not isinstance(self.body.ops[-1], ReturnOp)):
            raise VerifyException("Region does not end in return")
        if len(self.body.blocks[0].args) != len(self.func_type.inputs.data):
            raise VerifyException("Incorrect number of arguments")
        for arg, arg_type in zip(self.body.blocks[0].args,
                                 self.func_type.inputs.data):
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
        ret_op = self.body.ops[-1]
        if not isinstance(ret_op, ReturnOp):
            raise ValueError("Region does not end in a return")
        return ret_op.ret

    @staticmethod
    def from_function_type(func_type: FunctionType,
                           name: str | StringAttr | None = None):
        block = Block.from_arg_types(func_type.inputs.data)
        region = Region.from_block_list([block])
        if name is None:
            return DefineFunOp.create(result_types=[func_type],
                                      regions=[region])
        else:
            return DefineFunOp.build(result_types=[func_type],
                                     attributes={"name": name},
                                     regions=[region])

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(define-fun ", file=stream, end='')

        # Print the function name
        name: str
        if self.fun_name is not None:
            name = ctx.get_fresh_name(self.fun_name.data)
            ctx.value_to_name[self.ret] = name
        else:
            name = ctx.get_fresh_name(self.ret)
        print(f"{name} ", file=stream, end='')

        # Print the function arguments
        print("(", file=stream, end='')
        for idx, arg in enumerate(self.body.blocks[0].args):
            if idx != 0:
                print(" ", file=stream, end='')
            arg_name = ctx.get_fresh_name(arg)
            typ = arg.typ
            assert isinstance(typ, SMTLibSort)
            print(f"({arg_name} ", file=stream, end='')
            typ.print_sort_to_smtlib(stream)
            print(")", file=stream, end='')
        print(") ", file=stream, end='')

        # Print the function return type
        assert (len(self.func_type.outputs.data) == 1)
        ret_type = self.func_type.outputs.data[0]
        assert isinstance(ret_type, SMTLibSort)
        ret_type.print_sort_to_smtlib(stream)
        print("", file=stream)

        # Print the function body
        print("  ", file=stream, end='')
        ctx.print_expr_to_smtlib(self.return_val, stream)
        print(")", file=stream)


@irdl_op_definition
class ReturnOp(Operation):
    """The return operation of a function."""
    name = "smt.return"
    ret: Operand

    @staticmethod
    def from_ret_value(ret_value: SSAValue):
        return ReturnOp.create(operands=[ret_value])

    def verify_(self):
        parent = self.parent_op()
        if not isinstance(parent, DefineFunOp):
            raise ValueError("ReturnOp must be nested inside a DefineFunOp")
        if not self.ret.typ == parent.func_type.outputs.data[0]:
            raise ValueError("ReturnOp type mismatch with DefineFunOp")


@irdl_op_definition
class DeclareConstOp(Operation, SMTLibScriptOp):
    """Declare a constant value."""

    name = "smt.declare_const"
    res: OpResult

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        name = ctx.get_fresh_name(self.res)
        typ = self.res.typ
        assert isinstance(typ, SMTLibSort)
        print(f"(declare-const {name} ", file=stream, end='')
        typ.print_sort_to_smtlib(stream)
        print(")", file=stream)

    @classmethod
    def parse(cls: type[DeclareConstOp], result_types: list[Attribute],
              parser: BaseParser) -> DeclareConstOp:
        return DeclareConstOp.create(result_types=result_types)

    def print(self, printer: Printer):
        pass


@irdl_op_definition
class AssertOp(Operation, SMTLibScriptOp):
    """Assert that a boolean expression is true."""
    name = "smt.assert"
    op: Annotated[Operand, BoolType]

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(assert ", file=stream, end='')
        ctx.print_expr_to_smtlib(self.op, stream)
        print(")", file=stream)

    @classmethod
    def parse(cls: type[AssertOp], result_types: list[Attribute],
              parser: BaseParser) -> AssertOp:
        operand = parser.parse_operand()
        return AssertOp.create(operands=[operand])

    def print(self, printer: Printer):
        printer.print(" ", self.op)


@irdl_op_definition
class CheckSatOp(Operation, SMTLibScriptOp):
    """Check if the current set of assertions is satisfiable."""
    name = "smt.check_sat"

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        print("(check-sat)", file=stream)

    @classmethod
    def parse(cls: type[CheckSatOp], result_types: list[Attribute],
              parser: BaseParser) -> CheckSatOp:
        return CheckSatOp.create()

    def print(self, printer: Printer):
        pass


# Core operations

_OpT = TypeVar("_OpT", bound=Operation)


class BinaryBoolOp(Operation):
    """Base class for binary boolean operations."""
    res: Annotated[OpResult, BoolType]
    lhs: Annotated[Operand, BoolType]
    rhs: Annotated[Operand, BoolType]

    @classmethod
    def get(cls: type[_OpT], lhs: SSAValue, rhs: SSAValue) -> _OpT:
        return cls.create(result_types=[BoolType([])], operands=[lhs, rhs])

    @classmethod
    def parse(cls: type[_OpT], result_types: list[Attribute],
              parser: BaseParser) -> _OpT:
        lhs = parser.parse_operand()
        parser.parse_characters(",", "Expected `,`")
        rhs = parser.parse_operand()
        return cls.build(result_types=[BoolType([])], operands=[lhs, rhs])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.lhs)
        printer.print(", ")
        printer.print_ssa_value(self.rhs)


class BinaryTOp(Operation):
    """Base class for binary operations with boolean results."""
    res: Annotated[OpResult, BoolType]
    lhs: Operand
    rhs: Operand

    @classmethod
    def parse(cls: type[_OpT], result_types: list[Attribute],
              parser: BaseParser) -> _OpT:
        lhs = parser.parse_operand()
        parser.parse_characters(",", "Expected `,`")
        rhs = parser.parse_operand()
        return cls.build(result_types=[BoolType([])], operands=[lhs, rhs])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.lhs)
        printer.print(", ")
        printer.print_ssa_value(self.rhs)

    def verify_(self) -> None:
        if self.lhs.typ != self.rhs.typ:
            raise ValueError("Operands must have the same type")


@irdl_attr_definition
class BoolAttr(Data[bool]):
    """Boolean value."""

    name = "smt.bool_attr"

    @staticmethod
    def parse_parameter(parser: BaseParser) -> bool:
        val = parser.expect(parser.try_parse_bare_id,
                            "Expected 'true' or 'false'")
        if val.text == "true":
            return True
        if val.text == "false":
            return False
        raise ValueError("Expected 'true' or 'false'")

    @staticmethod
    def print_parameter(data: bool, printer: Printer) -> None:
        printer.print(str(data))


@irdl_op_definition
class ConstantBoolOp(Operation, SMTLibOp):
    """Boolean constant."""

    name = "smt.constant_bool"

    res: Annotated[OpResult, BoolType]
    value: OpAttr[BoolAttr]

    def print_expr_to_smtlib(self, stream: IOBase, ctx: SMTConversionCtx):
        if self.value.data:
            print("true", file=stream, end='')
        else:
            print("false", file=stream, end='')

    @staticmethod
    def from_bool(value: bool) -> ConstantBoolOp:
        return ConstantBoolOp.create(result_types=[BoolType([])],
                                     attributes={"value": BoolAttr(value)})

    @classmethod
    def parse(cls, result_types: list[Attribute],
              parser: BaseParser) -> ConstantBoolOp:
        val = parser.expect(parser.try_parse_bare_id,
                            "Expected 'true' or 'false'")
        if val.text == "true":
            return cls.from_bool(True)
        if val.text == "false":
            return cls.from_bool(False)
        raise ValueError("Expected 'true' or 'false'")

    def print(self, printer: Printer) -> None:
        if self.value.data:
            printer.print(" true")
        else:
            printer.print(" false")


@irdl_op_definition
class NotOp(Operation, SimpleSMTLibOp):
    """Boolean negation."""

    name = "smt.not"

    res: Annotated[OpResult, BoolType]
    arg: Annotated[Operand, BoolType]

    @classmethod
    def parse(cls, result_types: list[Attribute], parser: BaseParser) -> NotOp:
        val = parser.parse_operand()
        return cls.build(result_types=[BoolType([])], operands=[val])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.arg)

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
class DiscinctOp(BinaryTOp, SimpleSMTLibOp):
    """Distinctness."""

    name = "smt.distinct"

    def op_name(self) -> str:
        return "distinct"


@irdl_op_definition
class IteOp(Operation, SimpleSMTLibOp):
    """If-then-else."""

    name = "smt.ite"

    res: OpResult
    cond: Annotated[Operand, BoolType]
    true_val: Operand
    false_val: Operand

    @classmethod
    def parse(cls: type[IteOp], result_types: list[Attribute],
              parser: BaseParser) -> IteOp:
        cond = parser.parse_operand()
        parser.parse_characters(",", "Expected ','")
        true_val = parser.parse_operand()
        parser.parse_characters(",", "Expected ','")
        false_val = parser.parse_operand()
        return cls.create(result_types=[true_val.typ],
                          operands=[cond, true_val, false_val])

    def print(self, printer: Printer) -> None:
        printer.print(" ")
        printer.print_ssa_value(self.cond)
        printer.print(", ")
        printer.print_ssa_value(self.true_val)
        printer.print(", ")
        printer.print_ssa_value(self.false_val)

    def verify_(self) -> None:
        if not (self.res.typ == self.true_val.typ == self.false_val.typ):
            raise ValueError(
                "The result and both values must have the same type")

    def op_name(self) -> str:
        return "ite"


SMTDialect = Dialect([
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
    DiscinctOp,
    IteOp,
], [BoolType, BoolAttr])
