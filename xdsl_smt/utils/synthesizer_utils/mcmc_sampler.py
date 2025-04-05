import argparse
from typing import Callable

from xdsl.context import MLContext
from xdsl.dialects.builtin import i1, IntegerAttr, FunctionType
from xdsl.parser import Parser

from xdsl.utils.exceptions import VerifyException
from xdsl_smt.utils.synthesizer_utils.compare_result import CompareResult
from xdsl_smt.utils.synthesizer_utils.mutation_program import MutationProgram
from xdsl_smt.utils.synthesizer_utils.synthesizer_context import SynthesizerContext
from xdsl_smt.utils.synthesizer_utils.random import Random
from xdsl_smt.dialects.transfer import (
    AbstractValueType,
    TransIntegerType,
    GetOp,
    MakeOp,
    GetAllOnesOp,
    Constant,
    AndOp,
    CmpOp,
)
import xdsl.dialects.arith as arith
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Operation, OpResult
import sys as sys


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
    last_make_op: MakeOp
    current: MutationProgram
    proposed: MutationProgram | None
    current_cmp: CompareResult
    context: SynthesizerContext
    random: Random
    compute_cost: Callable[[CompareResult], float]
    is_cond: bool

    def __init__(
        self,
        func: FuncOp,
        context: SynthesizerContext,
        compute_cost: Callable[[CompareResult], float],
        length: int,
        reset: bool = True,
        random_init_program: bool = True,
        is_cond: bool = False,
        init_cmp_res: CompareResult = CompareResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 4),
    ):
        self.is_cond = is_cond
        if is_cond:
            cond_type = FunctionType.from_lists(
                func.function_type.inputs, [i1]  # pyright: ignore [reportArgumentType]
            )
            func = FuncOp("cond", cond_type)

        if reset:
            self.current = self.construct_init_program(func, length)
        else:
            self.current = MutationProgram(func.clone())
        self.proposed = None
        self.current_cmp = init_cmp_res
        self.context = context
        self.compute_cost = compute_cost
        self.random = context.get_random_class()
        if random_init_program:
            self.reset_to_random_prog(length)

    def compute_current_cost(self):
        return self.compute_cost(self.current_cmp)

    def get_current(self):
        return self.current.func

    def get_proposed(self):
        if self.proposed is None:
            return None
        return self.proposed.func

    def accept_proposed(self, proposed_cmp: CompareResult):
        assert self.proposed is not None
        self.current = self.proposed
        self.current_cmp = proposed_cmp
        self.proposed = None

    def reject_proposed(self):
        self.proposed = None

    @staticmethod
    def is_i1_op(op: Operation):
        return op.results[0].type == i1

    @staticmethod
    def is_int_op(op: Operation):
        return isinstance(op.results[0].type, TransIntegerType)

    def replace_entire_operation(self, prog: MutationProgram, idx: int) -> float:
        """
        Random pick an operation and replace it with a new one
        """
        old_op = prog.ops[idx]
        int_operands, _ = prog.get_valid_int_operands(idx)
        bool_operands, _ = prog.get_valid_bool_operands(idx)

        new_op = None
        while new_op is None:
            if MCMCSampler.is_i1_op(old_op):  # bool
                new_op = self.context.get_random_i1_op(int_operands, bool_operands)
            elif MCMCSampler.is_int_op(old_op):  # integer
                new_op = self.context.get_random_int_op(int_operands, bool_operands)
            else:
                raise VerifyException("Unexpected result type {}".format(old_op))
        # print(old_op)
        # print(new_op)
        prog.replace_operation(old_op, new_op)
        # print(prog.func)
        return 1

    def replace_operand(self, prog: MutationProgram, idx: int) -> float:
        op = prog.ops[idx]
        int_operands, _ = prog.get_valid_int_operands(idx)
        bool_operands, _ = prog.get_valid_bool_operands(idx)

        success = False
        while not success:
            success = self.context.replace_operand(op, int_operands, bool_operands)

        return 1

    # def replace_make_operand(self, ops: list[Operation], make_op_idx: int) -> float:
    #     idx = make_op_idx
    #     op = ops[idx]
    #     assert isinstance(op, MakeOp)
    #
    #     int_operands, _ = self.get_valid_int_operands(ops, idx)
    #     ith = self.random.randint(0, len(op.operands) - 1)
    #     assert isinstance(op.operands[ith].type, TransIntegerType)
    #     new_operand = self.random.choice(int_operands)
    #     op.operands[ith] = new_operand
    #     return 1

    def construct_init_program(self, _func: FuncOp, length: int):
        func = _func.clone()
        block = func.body.block
        for op in block.ops:
            block.detach_op(op)

        # Part I: GetOp
        for arg in block.args:
            if isinstance(arg.type, AbstractValueType):
                for i, field_type in enumerate(arg.type.get_fields()):
                    op = GetOp(arg, i)
                    block.add_op(op)

        assert isinstance(block.last_op, GetOp)
        tmp_int_ssavalue = block.last_op.results[0]

        # Part II: Constants
        true: arith.ConstantOp = arith.ConstantOp(
            IntegerAttr.from_int_and_width(1, 1), i1
        )
        false: arith.ConstantOp = arith.ConstantOp(
            IntegerAttr.from_int_and_width(0, 1), i1
        )
        all_ones = GetAllOnesOp(tmp_int_ssavalue)
        zero = Constant(tmp_int_ssavalue, 0)
        one = Constant(tmp_int_ssavalue, 1)
        block.add_op(true)
        block.add_op(false)
        block.add_op(zero)
        block.add_op(one)
        block.add_op(all_ones)

        if not self.is_cond:
            # Part III: Main Body
            last_int_op = block.last_op
            for i in range(length):
                if i % 4 == 0:
                    nop_bool = CmpOp(tmp_int_ssavalue, tmp_int_ssavalue, 0)
                    block.add_op(nop_bool)
                elif i % 4 == 1:
                    last_int_op = AndOp(tmp_int_ssavalue, tmp_int_ssavalue)
                    block.add_op(last_int_op)
                elif i % 4 == 2:
                    last_int_op = AndOp(tmp_int_ssavalue, tmp_int_ssavalue)
                    block.add_op(last_int_op)
                else:
                    last_int_op = AndOp(tmp_int_ssavalue, tmp_int_ssavalue)
                    block.add_op(last_int_op)

            # Part IV: MakeOp
            output = list(func.function_type.outputs)[0]
            assert isinstance(output, AbstractValueType)
            operands: list[OpResult] = []
            for i, field_type in enumerate(output.get_fields()):
                assert isinstance(field_type, TransIntegerType)
                assert last_int_op is not None
                operands.append(last_int_op.results[0])
                while True:
                    last_int_op = last_int_op.prev_op
                    assert last_int_op is not None
                    if MCMCSampler.is_int_op(last_int_op):
                        break

            return_val = MakeOp(operands)
            block.add_op(return_val)

        else:
            # Part III: Main Body
            last_bool_op = true
            for i in range(length):
                if i % 4 == 0:
                    last_int_op = AndOp(tmp_int_ssavalue, tmp_int_ssavalue)
                    block.add_op(last_int_op)
                else:
                    last_bool_op = CmpOp(tmp_int_ssavalue, tmp_int_ssavalue, 0)
                    block.add_op(last_bool_op)

            return_val = last_bool_op.results[0]

        # Part V: Return
        block.add_op(ReturnOp(return_val))

        return MutationProgram(func)

    def sample_next(self) -> float:
        """
        Sample the next program.
        Return the new program with the proposal ratio.
        """
        self.proposed = self.current.clone()

        # return_op = self.proposed.body.block.last_op
        # assert isinstance(return_op, Return)
        # last_make_op = return_op.operands[0].owner
        # assert isinstance(last_make_op, MakeOp)

        live_ops = self.proposed.get_modifiable_operations()
        live_op_indices = [_[1] for _ in live_ops]

        sample_mode = self.random.random()
        if (
            sample_mode < 0.3 and live_op_indices
        ):  # replace an operation with a new operation
            idx = self.random.choice(live_op_indices)
            ratio = self.replace_entire_operation(self.proposed, idx)
        elif sample_mode < 1 and live_op_indices:  # replace an operand in an operation
            idx = self.random.choice(live_op_indices)
            ratio = self.replace_operand(self.proposed, idx)
        # elif sample_mode < 1:
        #     # replace an operand in makeOp
        #     ratio = self.replace_make_operand(ops, len(ops) - 2)
        else:
            ratio = 1

        return ratio

    def reset_to_random_prog(self, length: int):
        # Part III-2: Random reset main body
        total_ops_len = len(self.current.ops)
        """
        Only modify ops in the main body
        """
        for i in range(total_ops_len):
            if not MutationProgram.not_in_main_body(self.current.ops[i]):
                self.replace_entire_operation(self.current, i)
