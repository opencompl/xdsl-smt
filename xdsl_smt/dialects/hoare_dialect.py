from xdsl.ir import Dialect, Operation, Region
from xdsl.irdl import (
    operand_def,
    region_def,
    irdl_op_definition,
    IRDLOperation,
    Operand,
)
from xdsl.utils.exceptions import VerifyException

from xdsl.traits import IsTerminator
from xdsl.dialects.func import FuncOp

from .smt_dialect import BoolType


@irdl_op_definition
class YieldOp(IRDLOperation):
    name = "hoare.yield"

    traits = frozenset([IsTerminator()])

    def __init__(self, ret: Operand):
        super().__init__(operands=[ret])

    ret: Operand = operand_def()


@irdl_op_definition
class RequiresOp(IRDLOperation):
    name = "hoare.requires"

    reg: Region = region_def("single_block")

    @property
    def yield_op(self) -> YieldOp:
        res = self.reg.block.last_op
        assert isinstance(res, YieldOp), "Should have been verified by the invariant"
        return res

    def verify_(self):
        if not isinstance((yield_op := self.reg.block.last_op), YieldOp):
            raise ValueError("Requires region must end with a yield operation")
        if not isinstance(yield_op.ret.type, BoolType):
            raise VerifyException(f"{self.name} must yield a boolean")
        if not isinstance((parent := self.parent_op()), FuncOp):
            raise VerifyException(f"{self.name} must be nested in a function")
        if len(self.reg.block.args) != len(parent.body.blocks[0].args):
            raise VerifyException(
                f"{self.name} region must have the same "
                "number of arguments as the function"
            )

    def __init__(self, reg: Region):
        super().__init__(regions=[reg])


@irdl_op_definition
class EnsuresOp(IRDLOperation):
    name = "hoare.ensures"

    reg: Region = region_def("single_block")

    @property
    def yield_op(self) -> Operation:
        res = self.reg.block.last_op
        assert res is not None, "Should have been verified by the invariant"
        return res

    def verify_(self):
        if not isinstance((yield_op := self.reg.block.last_op), YieldOp):
            raise VerifyException("Ensures region must end with a yield operation")
        if not isinstance(yield_op.ret.type, BoolType):
            raise VerifyException(f"{self.name} must yield a boolean")
        if not isinstance((parent := self.parent_op()), FuncOp):
            raise VerifyException(f"{self.name} must be nested in a function")
        if len(self.reg.block.args) != (
            len(parent.function_type.inputs) + len(parent.function_type.outputs)
        ):
            raise VerifyException(
                f"{self.name} region must have arguments for the "
                "function inputs and outputs"
            )

    def __init__(self, reg: Region):
        super().__init__(regions=[reg])


Hoare = Dialect(
    "hoare",
    [
        YieldOp,
        RequiresOp,
        EnsuresOp,
    ],
    [],
)
