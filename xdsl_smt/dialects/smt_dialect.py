from __future__ import annotations

from typing import Sequence, TypeVar, IO
from dataclasses import dataclass

from xdsl import traits
from xdsl.traits import IsTerminator, HasCanonicalizationPatternsTrait
from xdsl.irdl import (
    opt_attr_def,
    operand_def,
    result_def,
    var_result_def,
    region_def,
    var_operand_def,
    irdl_op_definition,
    Operand,
    IRDLOperation,
    traits_def,
)
from xdsl.ir import (
    Block,
    Dialect,
    OpResult,
    Operation,
    Attribute,
    SSAValue,
    Region,
    OpTraits,
)
from xdsl.dialects.builtin import FunctionType, StringAttr
from xdsl.dialects.smt import (
    AndOp,
    BoolType,
    ImpliesOp,
    OrOp,
    XOrOp,
    NotOp,
    ConstantBoolOp,
    YieldOp,
    ForallOp,
    ExistsOp,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.pattern_rewriter import RewritePattern

from xdsl_smt.traits.effects import Pure

from xdsl_smt.traits.smt_printer import (
    SMTLibOp,
    SMTLibScriptOp,
    SimpleSMTLibOp,
    SMTConversionCtx,
    SimpleSMTLibOpTrait,
    SMTLibOpTrait,
)


class QuantifierCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt import (
            QuantifierCanonicalizationPattern,
        )

        return (QuantifierCanonicalizationPattern(),)


@dataclass(frozen=True)
class QuantifierSMTLibOpTrait(SMTLibOpTrait):
    """Trait for quantifier operations that prints them in SMT-LIB format."""

    name: str

    def print_expr_to_smtlib(
        self, op: Operation, stream: IO[str], ctx: SMTConversionCtx
    ) -> None:
        assert isinstance(op, ForallOp) or isinstance(op, ExistsOp)
        print(f"({self.name} (", file=stream, end="")
        for idx, param in enumerate(op.body.block.args):
            param_name = ctx.get_fresh_name(param)
            if idx != 0:
                print(" ", file=stream, end="")
            print(f"({param_name} ", file=stream, end="")
            ctx.print_sort_to_smtlib(param.type, stream)
            print(")", file=stream, end="")
        print(") ", file=stream, end="")
        ctx.print_expr_to_smtlib(op.returned_value, stream)
        print(")", file=stream, end="")


ForallOp_traits = ForallOp.traits
ForallOp.traits = OpTraits(
    lambda: (QuantifierCanonicalizationPatterns(), QuantifierSMTLibOpTrait("forall"))
)

ExistsOp_traits = ExistsOp.traits
ExistsOp.traits = OpTraits(
    lambda: (QuantifierCanonicalizationPatterns(), QuantifierSMTLibOpTrait("exists"))
)


@irdl_op_definition
class CallOp(IRDLOperation, Pure, SMTLibOp):
    """Call to an SMT function."""

    name = "smt.call"

    res = var_result_def()
    func = operand_def(FunctionType)
    args = var_operand_def()

    traits = traits_def(traits.Pure())

    def __init__(self, func: Operand, args: Sequence[Operand | Operation]):
        if not isinstance(func.type, FunctionType):
            raise Exception("Expected function type, got ", func.type)
        super().__init__(
            operands=[func, args],
            result_types=[func.type.outputs.data],
        )

    @staticmethod
    def get(func: Operand, args: Sequence[Operand | Operation]) -> CallOp:
        if not isinstance(func.type, FunctionType):
            raise Exception("Expected function type, got ", func.type)
        return CallOp.build(
            operands=[func, args],
            result_types=[func.type.outputs.data[0]],
        )

    def verify_(self) -> None:
        assert isinstance(self.func.type, FunctionType)
        if len(self.args) != len(self.func.type.inputs.data):
            raise VerifyException("Incorrect number of arguments")
        for arg, arg_type in zip(self.args, self.func.type.inputs.data):
            if arg.type != arg_type:
                raise VerifyException("Incorrect argument type")
        if len(self.func.type.outputs.data) != 1:
            raise VerifyException("Incorrect number of return values")
        if tuple(res.type for res in self.res) != self.func.type.outputs.data:
            raise VerifyException("Incorrect return type")

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        # In the case there are no arguments, we do not print the outer parentheses
        if not self.args:
            ctx.print_expr_to_smtlib(self.func, stream)
            return

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
class DeclareFunOp(IRDLOperation, SMTLibScriptOp):
    """
    Declare an uninterpreted function.
    Unlike SMT-LIB, to ease the manipulation of functions, functions may have more
    than one return value.
    Functions with more than one return value should be converted to multiple
    functions or to a function with a tuple return type before conversion to SMT-LIB.
    """

    name = "smt.declare_fun"

    fun_name: StringAttr | None = opt_attr_def(StringAttr)
    ret: OpResult = result_def(FunctionType)

    def __init__(
        self, func_type: FunctionType, name: str | StringAttr | None = None
    ) -> None:
        if isinstance(name, str):
            name = StringAttr(name)
        attributes = {"fun_name": name} if name is not None else {}
        super().__init__(
            attributes=attributes,
            result_types=[func_type],
        )

    @property
    def func_type(self) -> FunctionType:
        """Get the function type of this operation."""
        if not isinstance(self.ret.type, FunctionType):
            raise VerifyException("Incorrect return type")
        return self.ret.type

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        print("(declare-fun ", file=stream, end="")

        # Print the function name
        name: str
        if self.fun_name is not None:
            name = ctx.get_fresh_name(self.fun_name.data)
            ctx.value_to_name[self.ret] = name
        else:
            name = ctx.get_fresh_name(self.ret)
        print(f"{name} ", file=stream, end="")

        # Print the function arguments
        for idx, typ in enumerate(self.func_type.inputs):
            if idx != 0:
                print(" ", file=stream, end="")
            print("(", file=stream, end="")
            ctx.print_sort_to_smtlib(typ, stream)
            print(")", file=stream, end="")

        # Print the function return type
        assert len(self.func_type.outputs.data) == 1
        ret_type = self.func_type.outputs.data[0]
        ctx.print_sort_to_smtlib(ret_type, stream)
        print(")", file=stream)


@irdl_op_definition
class DefineFunOp(IRDLOperation, SMTLibScriptOp):
    """
    Define a function.
    Unlike SMT-LIB, to ease the manipulation of functions, functions may have more
    than one return value.
    Functions with more than one return value should be converted to multiple
    functions or to a function with a tuple return type before conversion to SMT-LIB.
    """

    name = "smt.define_fun"

    fun_name: StringAttr | None = opt_attr_def(StringAttr)
    ret: OpResult = result_def(FunctionType)
    body: Region = region_def("single_block")

    def __init__(self, region: Region, name: str | StringAttr | None = None) -> None:
        """
        Create a new function given its body and name.
        The body is expected to have a single block terminated by an `smt.return`.
        """
        operand_types = region.block.arg_types
        if not isinstance(terminator := region.block.last_op, ReturnOp):
            raise Exception(f"In {self.name} constructor: Region must end in a return")
        result_types = tuple(res.type for res in terminator.ret)
        if isinstance(name, str):
            name = StringAttr(name)
        attributes = {"fun_name": name} if name is not None else {}
        super().__init__(
            attributes=attributes,
            result_types=[FunctionType.from_lists(operand_types, result_types)],
            regions=[region],
        )

    def verify_(self) -> None:
        if len(self.body.ops) == 0 or not isinstance(self.body.block.last_op, ReturnOp):
            raise VerifyException("Region does not end in return")
        if len(self.body.blocks[0].args) != len(self.func_type.inputs.data):
            raise VerifyException("Incorrect number of arguments")
        for arg, arg_type in zip(self.body.blocks[0].args, self.func_type.inputs.data):
            if arg.type != arg_type:
                raise VerifyException("Incorrect argument type")
        for arg, arg_type in zip(self.body.blocks[0].args, self.func_type.inputs.data):
            if arg.type != arg_type:
                raise VerifyException("Incorrect argument type")
        for ret, ret_type in zip(self.return_values, self.func_type.outputs.data):
            if ret.type != ret_type:
                raise VerifyException("Incorrect return type")

    @property
    def func_type(self) -> FunctionType:
        """Get the function type of this operation."""
        if not isinstance(self.ret.type, FunctionType):
            raise VerifyException("Incorrect return type")
        return self.ret.type

    @property
    def return_values(self) -> Sequence[SSAValue]:
        """Get the return values of this operation."""
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
                result_types=[func_type],
                attributes={"fun_name": name},
                regions=[region],
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
            typ = arg.type
            print(f"({arg_name} ", file=stream, end="")
            ctx.print_sort_to_smtlib(typ, stream)
            print(")", file=stream, end="")
        print(") ", file=stream, end="")

        # Print the function return type
        assert len(self.func_type.outputs.data) == 1
        ret_type = self.func_type.outputs.data[0]
        ctx.print_sort_to_smtlib(ret_type, stream)
        print("", file=stream)

        # Print the function body
        print("  ", file=stream, end="")
        if len(self.return_values) != 1:
            raise Exception(
                "Functions with multiple return values cannot be converted to SMT-LIB"
            )
        ctx.print_expr_to_smtlib(self.return_values[0], stream, identation="  ")
        print(")", file=stream)


@irdl_op_definition
class ReturnOp(IRDLOperation):
    """The return operation of a function."""

    name = "smt.return"
    ret = var_operand_def()

    traits = traits_def(IsTerminator())

    def __init__(self, operand: SSAValue | Sequence[SSAValue]):
        super().__init__(operands=[operand])

    def verify_(self):
        parent = self.parent_op()
        if not isinstance(parent, DefineFunOp):
            raise VerifyException("ReturnOp must be nested inside a DefineFunOp")
        for ret, ret_type in zip(self.ret, parent.func_type.outputs.data):
            if ret.type != ret_type:
                raise VerifyException("Incorrect return type")


@irdl_op_definition
class DeclareConstOp(IRDLOperation, SMTLibScriptOp):
    """Declare a constant value."""

    name = "smt.declare_const"
    res: OpResult = result_def()

    def __init__(self, type: Attribute):
        super().__init__(result_types=[type])

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        name = ctx.get_fresh_name(self.res)
        typ = self.res.type
        print(f"(declare-const {name} ", file=stream, end="")
        ctx.print_sort_to_smtlib(typ, stream)
        print(")", file=stream)


@irdl_op_definition
class AssertOp(IRDLOperation, SMTLibScriptOp):
    """Assert that a boolean expression is true."""

    name = "smt.assert"
    op: Operand = operand_def(BoolType)

    def __init__(self, operand: SSAValue):
        super().__init__(operands=[operand])

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        print("(assert ", file=stream, end="")
        ctx.print_expr_to_smtlib(self.op, stream, identation="  ")
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

    res: OpResult = result_def(BoolType)
    lhs: Operand = operand_def(BoolType)
    rhs: Operand = operand_def(BoolType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[BoolType()], operands=[lhs, rhs])

    @classmethod
    def get(cls: type[_OpT], lhs: SSAValue, rhs: SSAValue) -> _OpT:
        return cls.create(result_types=[BoolType()], operands=[lhs, rhs])


class BinaryTOp(IRDLOperation, Pure):
    """Base class for binary operations with boolean results."""

    res: OpResult = result_def(BoolType)
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[BoolType()], operands=[lhs, rhs])

    @classmethod
    def get(cls: type[_OpT], lhs: SSAValue, rhs: SSAValue) -> _OpT:
        return cls.create(result_types=[BoolType()], operands=[lhs, rhs])

    def verify_(self) -> None:
        if self.lhs.type != self.rhs.type:
            raise VerifyException("Operands must have the same type")


class ConstantBoolOpPrinter(SMTLibOpTrait):
    def print_expr_to_smtlib(
        self, op: Operation, stream: IO[str], ctx: SMTConversionCtx
    ) -> None:
        assert isinstance(op, ConstantBoolOp)
        if op.value:
            print("true", file=stream, end="")
        else:
            print("false", file=stream, end="")


ConstantBoolOp.traits.add_trait(ConstantBoolOpPrinter())


class NotCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt import (
            NotCanonicalizationPattern,
        )

        return (NotCanonicalizationPattern(),)


NotOp.traits.add_trait(SimpleSMTLibOpTrait("not"))
NotOp.traits.add_trait(NotCanonicalizationPatterns())


class ImpliesCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt import (
            ImpliesCanonicalizationPattern,
        )

        return (ImpliesCanonicalizationPattern(),)


ImpliesOp.traits.add_trait(SimpleSMTLibOpTrait("=>"))
ImpliesOp.traits.add_trait(ImpliesCanonicalizationPatterns())


class AndCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt import (
            AndCanonicalizationPattern,
        )

        return (AndCanonicalizationPattern(),)


AndOp_traits = AndOp.traits
AndOp.traits = OpTraits(
    lambda: (*AndOp_traits, SimpleSMTLibOpTrait("and"), AndCanonicalizationPatterns())
)


class OrCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt import (
            OrCanonicalizationPattern,
        )

        return (OrCanonicalizationPattern(),)


OrOp_traits = OrOp.traits
OrOp.traits = OpTraits(
    lambda: (*OrOp_traits, SimpleSMTLibOpTrait("or"), OrCanonicalizationPatterns())
)


class XOrCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt import (
            XOrCanonicalizationPattern,
        )

        return (XOrCanonicalizationPattern(),)


XOrOp_traits = XOrOp.traits
XOrOp.traits = OpTraits(
    lambda: (*XOrOp_traits, SimpleSMTLibOpTrait("xor"), XOrCanonicalizationPatterns())
)


class EqCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt import (
            EqCanonicalizationPattern,
        )

        return (EqCanonicalizationPattern(),)


@irdl_op_definition
class EqOp(BinaryTOp, SimpleSMTLibOp):
    """Equality."""

    name = "smt.eq"

    traits = traits_def(traits.Pure(), EqCanonicalizationPatterns())

    def op_name(self) -> str:
        return "="


class DistinctCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt import (
            DistinctCanonicalizationPattern,
        )

        return (DistinctCanonicalizationPattern(),)


@irdl_op_definition
class DistinctOp(BinaryTOp, SimpleSMTLibOp):
    """Distinctness."""

    name = "smt.distinct"

    traits = traits_def(traits.Pure(), DistinctCanonicalizationPatterns())

    def op_name(self) -> str:
        return "distinct"


class IteCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt import (
            IteCanonicalizationPattern,
            ItePairsCanonicalizationPattern,
            IteMergePattern,
        )

        return (
            IteCanonicalizationPattern(),
            ItePairsCanonicalizationPattern(),
            IteMergePattern(),
        )


@irdl_op_definition
class IteOp(IRDLOperation, Pure, SimpleSMTLibOp):
    """If-then-else."""

    name = "smt.ite"

    res: OpResult = result_def()
    cond: Operand = operand_def(BoolType)
    true_val: Operand = operand_def()
    false_val: Operand = operand_def()

    traits = traits_def(traits.Pure(), IteCanonicalizationPatterns())

    def __init__(self, cond: SSAValue, true_val: SSAValue, false_val: SSAValue):
        super().__init__(
            result_types=[true_val.type], operands=[cond, true_val, false_val]
        )

    def verify_(self) -> None:
        if not (self.res.type == self.true_val.type == self.false_val.type):
            raise VerifyException("The result and both values must have the same type")

    def op_name(self) -> str:
        return "ite"


@irdl_op_definition
class EvalOp(IRDLOperation, SMTLibScriptOp):
    """Eval an expression."""

    name = "smt.eval"
    expr: Operand = operand_def()

    def __init__(self, operand: SSAValue):
        super().__init__(operands=[operand])

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx):
        print("(eval ", file=stream, end="")
        ctx.print_expr_to_smtlib(self.expr, stream, identation="  ")
        print(")", file=stream)

    @staticmethod
    def get(arg: Operation | SSAValue) -> EvalOp:
        return EvalOp.build(operands=[arg])


SMTDialect = Dialect(
    "smt",
    [
        YieldOp,
        ForallOp,
        ExistsOp,
        CallOp,
        DefineFunOp,
        ReturnOp,
        DeclareConstOp,
        DeclareFunOp,
        AssertOp,
        CheckSatOp,
        ConstantBoolOp,
        NotOp,
        ImpliesOp,
        AndOp,
        OrOp,
        XOrOp,
        EqOp,
        DistinctOp,
        IteOp,
        EvalOp,
    ],
    [BoolType],
)
