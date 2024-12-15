import argparse

from xdsl.context import MLContext
from xdsl.parser import Parser

from xdsl.utils.exceptions import VerifyException
from xdsl_smt.dialects import transfer
from xdsl.dialects import arith
from xdsl_smt.dialects.transfer import (
    AbstractValueType,
    TransIntegerType,
    GetOp,
    SelectOp,
    AndOp,
    OrOp,
    XorOp,
    CmpOp,
    MakeOp,
)
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    i1,
    IndexType,
)
from xdsl.dialects.func import FuncOp, Return
from xdsl.ir import Operation, OpResult
from xdsl_smt.semantics.transfer_semantics import (
    CmpOpSemantics,
)
import sys as sys
import random


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "transfer_functions", type=str, nargs="?", help="path to the transfer functions"
    )


def parse_file(ctx: MLContext, file: str | None) -> Operation:
    if file is None:
        f = sys.stdin
        file = "<stdin>"
    else:
        f = open(file)

    parser = Parser(ctx, f.read(), file)
    module = parser.parse_op()
    return module


class MCMCSampler:
    def __init__(self, func: FuncOp, length: int):
        MCMCSampler.construct_init_program(func, length)
        self.func = func

    def get_func(self):
        return self.func

    @staticmethod
    def get_valid_bool_operands(
        ops: list[Operation], x: int
    ) -> tuple[list[OpResult], int]:
        """
        Get operations that before ops[x] so that can serve as operands
        """
        bool_ops = [
            result for op in ops[:x] for result in op.results if result.type == i1
        ]
        bool_count = len(bool_ops)
        assert bool_count > 0
        return bool_ops, bool_count

    @staticmethod
    def get_valid_int_operands(
        ops: list[Operation], x: int
    ) -> tuple[list[OpResult], int]:
        """
        Get operations that before ops[x] so that can serve as operands
        """
        int_ops = [
            result
            for op in ops[:x]
            for result in op.results
            if isinstance(op.results[0].type, TransIntegerType)
        ]
        int_count = len(int_ops)
        assert int_count > 0
        return int_ops, int_count

    @staticmethod
    def replace_entire_operation(
        ops: list[Operation],
    ) -> tuple[Operation, Operation, float]:
        """
        Random pick an operation and replace it with a new one
        """
        modifiable_indices = range(6, len(ops) - 2)

        idx = random.choice(modifiable_indices)
        old_op = ops[idx]
        new_op = None

        int_operands, num_int_operands = MCMCSampler.get_valid_int_operands(ops, idx)
        bool_operands, num_bool_operands = MCMCSampler.get_valid_bool_operands(ops, idx)

        def calculate_operand_prob(op: Operation) -> int:
            ret = 1
            for operand in op.operands:
                if operand.type == IntegerType(4):
                    ret = ret * num_int_operands
                elif operand.type == i1:
                    ret = ret * num_bool_operands
            return ret

        if old_op.results[0].type == i1:  # bool
            candidate = [arith.AndI.name, arith.OrI.name, CmpOp.name]
            if old_op.name in candidate:
                candidate.remove(old_op.name)
            opcode = random.choice(candidate)
            op1 = random.choice(bool_operands)
            op2 = random.choice(bool_operands)
            if opcode == arith.AndI.name:
                new_op = arith.AndI(op1, op2)
            elif opcode == arith.OrI.name:
                new_op = arith.OrI(op1, op2)
            elif opcode == CmpOp.name:
                predicate = random.randrange(len(CmpOpSemantics.new_ops))
                int_op1 = random.choice(int_operands)
                int_op2 = random.choice(int_operands)
                new_op = CmpOp(int_op1, int_op2, predicate)
            else:
                assert False

            forward_prob = calculate_operand_prob(old_op)
            backward_prob = calculate_operand_prob(new_op)

        elif isinstance(old_op.results[0].type, TransIntegerType):  # integer
            candidate = [AndOp.name, OrOp.name, XorOp.name, SelectOp.name]
            if old_op.name in candidate:
                candidate.remove(old_op.name)
            opcode = random.choice(candidate)
            op1 = random.choice(int_operands)
            op2 = random.choice(int_operands)
            if opcode == AndOp.name:
                new_op = AndOp(op1, op2)
            elif opcode == OrOp.name:
                new_op = OrOp(op1, op2)
            elif opcode == XorOp.name:
                new_op = XorOp(op1, op2)
            elif opcode == SelectOp.name:
                cond = random.choice(bool_operands)
                new_op = SelectOp(cond, op1, op2)
            else:
                assert False

            forward_prob = calculate_operand_prob(old_op)
            backward_prob = calculate_operand_prob(new_op)

        else:
            raise VerifyException(
                "Unexpected result type {}".format(old_op.results[0].type)
            )

        return old_op, new_op, backward_prob / forward_prob

    @staticmethod
    def replace_operand(ops: list[Operation]) -> float:
        modifiable_indices = [
            i
            for i, op in enumerate(ops[6:-1], start=6)
            if op.operands and not isinstance(op, transfer.Constant)
        ]
        assert modifiable_indices
        idx = random.choice(modifiable_indices)
        op = ops[idx]
        int_operands, _ = MCMCSampler.get_valid_int_operands(ops, idx)
        bool_operands, _ = MCMCSampler.get_valid_bool_operands(ops, idx)

        ith = random.randrange(len(op.operands))
        if op.operands[ith].type == i1:
            new_operand = random.choice(bool_operands)
        elif isinstance(op.operands[ith].type, TransIntegerType):
            new_operand = random.choice(int_operands)
        else:
            raise VerifyException(
                "Unexpected operand type {}".format(op.operands[ith].type)
            )

        # print(f'modifying op: {idx}' )
        # print(f'old operand: {op.operands[ith]}')
        # print(f'new operand: {new_operand}')
        op.operands[ith] = new_operand
        return 1

    @staticmethod
    def construct_init_program(func: FuncOp, length: int):
        block = func.body.block

        for op in block.ops:
            block.detach_op(op)

        # Part I: Constants
        true: arith.Constant = arith.Constant(IntegerAttr.from_int_and_width(1, 1), i1)
        false: arith.Constant = arith.Constant(IntegerAttr.from_int_and_width(0, 1), i1)
        # zero: Constant = Constant(IntegerAttr.from_int_and_width(0, 4), IntegerType(4))
        # one: Constant = Constant(IntegerAttr.from_int_and_width(1, 4), IntegerType(4))
        block.add_op(true)
        block.add_op(false)
        # block.add_op(zero)
        # block.add_op(one)

        # Part II: GetOp
        for arg in block.args:
            if isinstance(arg.type, AbstractValueType):
                for i, field_type in enumerate(arg.type.get_fields()):
                    op = GetOp(
                        operands=[arg],
                        attributes={"index": IntegerAttr(i, IndexType())},
                        result_types=[field_type],
                    )
                    block.add_op(op)

        # Part III: Main Body
        assert isinstance(block.last_op, GetOp)
        tmp_int_ssavalue = block.last_op.results[0]
        tmp_bool_ssavalue = true.results[0]
        for i in range(length // 2):
            # nop_bool = arith.Constant(IntegerAttr.from_int_and_width(1, 1), i1)
            # nop_int = transfer.Constant(tmp_int_ssavalue, 0)
            nop_bool = arith.AndI(tmp_bool_ssavalue, tmp_bool_ssavalue)
            nop_int = transfer.AndOp(tmp_int_ssavalue, tmp_bool_ssavalue)
            block.add_op(nop_bool)
            block.add_op(nop_int)

        # Part IV: MakeOp
        return_val: list[Operation] = []
        for output in func.function_type.outputs:
            assert isinstance(output, AbstractValueType)
            operands: list[OpResult] = []
            for i, field_type in enumerate(output.get_fields()):
                assert isinstance(field_type, TransIntegerType)
                operands.append(tmp_int_ssavalue)

            op = MakeOp(
                operands=[operands],
                result_types=MakeOp.infer_result_type(
                    [operand.type for operand in operands]
                ),
            )
            block.add_op(op)
            return_val.append(op)

        # Part V: Return
        block.add_op(Return(return_val[0]))
        return

    def sample_next(self) -> float:
        """
        Sample the next program.
        Return the new program with the proposal ratio.
        """
        ops = list(self.func.body.block.ops)

        sample_mode = random.randrange(2)
        if sample_mode == 0:
            # replace an operation with a new operation
            old_op, new_op, ratio = MCMCSampler.replace_entire_operation(ops)
            self.func.body.block.insert_op_before(new_op, old_op)
            if len(old_op.results) > 0 and len(new_op.results) > 0:
                old_op.results[0].replace_by(new_op.results[0])
            self.func.body.block.detach_op(old_op)

        elif sample_mode == 1:
            # replace an operand in an operand
            ratio = MCMCSampler.replace_operand(ops)

        elif sample_mode == 2:
            # todo: replace NOP with an operations
            ratio = 1
        else:
            # todo: replace an operations with NOP
            ratio = 1

        return ratio
