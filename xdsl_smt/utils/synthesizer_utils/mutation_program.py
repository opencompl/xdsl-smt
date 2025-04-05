from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Operation, SSAValue
import xdsl.dialects.arith as arith

from xdsl_smt.dialects.transfer import (
    GetOp,
    MakeOp,
    Constant,
    GetAllOnesOp,
    TransIntegerType,
)
from xdsl.dialects.builtin import (
    i1,
)


class MutationProgram:
    """
    Represents a program that is mutated during MCMC.

    Attributes:
        func (FuncOp): The mutation program.
        ops (list[Operation]): A list of operations within the function's body.

    The user should **manually** maintain the consistency between func and ops.
    """

    func: FuncOp
    ops: list[Operation]

    def __init__(self, func: FuncOp, ops: list[Operation] | None = None):
        if ops is None:
            ops = list(func.body.block.ops)
        self.func = func
        self.ops = ops

    def clone(self):
        new_func = self.func.clone()
        new_ops = list(new_func.body.block.ops)
        return MutationProgram(new_func, new_ops)

    @staticmethod
    def not_in_main_body(op: Operation):
        # filter out operations not belong to main body
        return (
            isinstance(op, Constant)
            or isinstance(op, arith.ConstantOp)
            or isinstance(op, GetAllOnesOp)
            or isinstance(op, GetOp)
            or isinstance(op, MakeOp)
            or isinstance(op, ReturnOp)
        )

    def get_modifiable_operations(
        self, only_live: bool = True
    ) -> list[tuple[Operation, int]]:
        """
        Get live operations when only_live = True, otherwise return all operations in the main body
        """
        modifiable_ops = list[tuple[Operation, int]]()
        live_set = set[Operation]()

        assert isinstance(self.ops[-1], ReturnOp)

        if isinstance(self.ops[-2], MakeOp):  # regular function
            last_make_op = self.ops[-2]
            for operand in last_make_op.operands:
                assert isinstance(operand.owner, Operation)
                live_set.add(operand.owner)
        else:  # condition
            assert not MutationProgram.not_in_main_body(self.ops[-2])
            live_set.add(self.ops[-2])

        for idx in range(len(self.ops) - 2, -1, -1):
            operation = self.ops[idx]
            if MutationProgram.not_in_main_body(operation):
                continue
            if only_live:
                if operation in live_set:
                    modifiable_ops.append((operation, idx))
                    for operand in operation.operands:
                        assert isinstance(operand.owner, Operation)
                        live_set.add(operand.owner)
            else:
                modifiable_ops.append((operation, idx))

        return modifiable_ops

    def replace_operation(self, old_op: Operation, new_op: Operation):
        """
        Replace the old_op with the given new operation.
        """
        block = self.func.body.block
        block.insert_op_before(new_op, old_op)
        if len(old_op.results) > 0 and len(new_op.results) > 0:
            old_op.results[0].replace_by(new_op.results[0])
        block.detach_op(old_op)
        self.ops = list(block.ops)

    def get_valid_bool_operands(self, x: int) -> tuple[list[SSAValue], int]:
        """
        Get bool operations that before ops[x] so that can serve as operands
        """
        bool_ops: list[SSAValue] = [
            result for op in self.ops[:x] for result in op.results if result.type == i1
        ]
        bool_count = len(bool_ops)
        return bool_ops, bool_count

    def get_valid_int_operands(self, x: int) -> tuple[list[SSAValue], int]:
        """
        Get int operations that before ops[x] so that can serve as operands
        """
        int_ops: list[SSAValue] = [
            result
            for op in self.ops[:x]
            for result in op.results
            if isinstance(result.type, TransIntegerType)
        ]
        int_count = len(int_ops)
        return int_ops, int_count
