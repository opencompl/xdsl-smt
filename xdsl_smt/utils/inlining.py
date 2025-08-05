from typing import Sequence

from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import SSAValue, BlockArgument
from xdsl.rewriter import InsertPoint, Rewriter


def inline_single_result_func(
    func: FuncOp, args: Sequence[SSAValue], insert_point: InsertPoint
) -> SSAValue:
    """
    Inline a single-result function at the current location.
    """
    assert len(func.function_type.outputs) == 1, "Function must have a single output."
    assert len(func.body.blocks) == 1, "Function must have a single block."

    block_copy = func.body.clone().block
    return_op = block_copy.last_op
    assert isinstance(return_op, ReturnOp), "Function must end with a return operation."
    return_value = return_op.operands[0]
    block_copy.erase_op(return_op)

    if return_value in block_copy.args:
        assert isinstance(return_value, BlockArgument)
        return_value = args[return_value.index]

    Rewriter.inline_block(block_copy, insert_point, args)
    return return_value
