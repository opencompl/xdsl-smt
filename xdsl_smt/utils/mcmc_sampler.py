import argparse

from xdsl.context import MLContext
from xdsl.dialects.builtin import i1, IntegerAttr
from xdsl.parser import Parser

from xdsl.utils.exceptions import VerifyException
from xdsl_smt.utils.compare_result import CompareResult
from xdsl_smt.utils.mutation_program import MutationProgram
from xdsl_smt.utils.synthesizer_context import SynthesizerContext
from xdsl_smt.utils.random import Random
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

    def __init__(
        self,
        func: FuncOp,
        context: SynthesizerContext,
        length: int,
        init_cost: float,
        reset: bool = True,
        random_init_program: bool = True,
        init_cmp_res: CompareResult = CompareResult(0, 0, 0, 0, 0, 0, 0, 0, 4),
    ):
        if reset:
            self.current = self.construct_init_program(func, length)
        else:
            self.current = MutationProgram(func.clone())
        self.proposed = None
        self.current_cmp = init_cmp_res
        self.context = context
        self.random = context.get_random_class()
        if random_init_program:
            self.reset_to_random_prog(length)

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

    def replace_entire_operation(self, prog: MutationProgram, idx: int) -> float:
        """
        Random pick an operation and replace it with a new one
        """
        old_op = prog.ops[idx]
        int_operands, _ = prog.get_valid_int_operands(idx)
        bool_operands, _ = prog.get_valid_bool_operands(idx)

        new_op = None
        while new_op is None:
            if old_op.results[0].type == i1:  # bool
                new_op = self.context.get_random_i1_op(int_operands, bool_operands)
            elif isinstance(old_op.results[0].type, TransIntegerType):  # integer
                new_op = self.context.get_random_int_op(int_operands, bool_operands)
            else:
                raise VerifyException("Unexpected result type {}".format(old_op))
        prog.replace_operation(old_op, new_op)
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

        # Part III: Main Body
        # tmp_bool_ssavalue = false_op.results[0]
        for i in range(length // 4):
            nop_bool = CmpOp(tmp_int_ssavalue, tmp_int_ssavalue, 0)
            nop_int1 = AndOp(tmp_int_ssavalue, tmp_int_ssavalue)
            nop_int2 = AndOp(tmp_int_ssavalue, tmp_int_ssavalue)
            nop_int3 = AndOp(tmp_int_ssavalue, tmp_int_ssavalue)
            block.add_op(nop_bool)
            block.add_op(nop_int1)
            block.add_op(nop_int2)
            block.add_op(nop_int3)

        last_int_op = block.last_op

        # Part IV: MakeOp
        return_val: list[Operation] = []
        for output in func.function_type.outputs:
            assert isinstance(output, AbstractValueType)
            operands: list[OpResult] = []
            for i, field_type in enumerate(output.get_fields()):
                assert isinstance(field_type, TransIntegerType)
                assert last_int_op is not None
                operands.append(last_int_op.results[0])
                assert last_int_op.prev_op is not None
                last_int_op = last_int_op.prev_op.prev_op

            op = MakeOp(operands)
            block.add_op(op)
            return_val.append(op)

        # Part V: Return
        block.add_op(ReturnOp(return_val[0]))

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
        the function in self.current should include:
          init_constant_values
          main body with number of operations (length)
          last make and return op

        As a result, total_ops_len = len(constant_values) + length + 2
        the last operation at the main body should have idea total_ops_len - 3
        so we iterate i 0 -> length, and replace the operation at total_ops_len - 3 - i
        """
        for i in range(length):
            self.replace_entire_operation(self.current, total_ops_len - 3 - i)
