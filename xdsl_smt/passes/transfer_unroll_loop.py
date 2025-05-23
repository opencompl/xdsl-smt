from dataclasses import dataclass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

from ..dialects import transfer
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import SSAValue
from xdsl.context import Context
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern


@dataclass(frozen=True)
class ConstRangeForOpPattern(RewritePattern):
    width: int

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: transfer.ConstRangeForOp, rewriter: PatternRewriter
    ) -> None:
        lb = op.lb.owner
        ub = op.ub.owner
        step = op.step.owner
        if isinstance(lb, transfer.Constant):
            lb_int = lb.value.value.data
        elif isinstance(lb, transfer.GetBitWidthOp):
            lb_int = self.width
        else:
            assert False and "loop lower bound has to be a constant"

        if isinstance(ub, transfer.Constant):
            ub_int = ub.value.value.data
        elif isinstance(ub, transfer.GetBitWidthOp):
            ub_int = self.width
        else:
            assert False and "loop upper bound has to be a constant"

        if isinstance(step, transfer.Constant):
            step_int = step.value.value.data
        elif isinstance(step, transfer.GetBitWidthOp):
            step_int = self.width
        else:
            assert False and "loop step has to be a constant"

        assert step_int != 0 and "step size should not be zero"
        if step_int > 0:
            assert (
                ub_int > lb_int
                and "the upper bound should be larger than the lower bound"
            )
        else:
            assert (
                ub_int < lb_int
                and "the upper bound should be smaller than the lower bound"
            )

        iter_args = [arg for arg in op.iter_args]
        iter_args_num = len(iter_args)

        indvar, *block_iter_args = op.body.block.args
        value_map: dict[SSAValue, SSAValue] = {}

        value_map[indvar] = op.lb
        for i in range(iter_args_num):
            value_map[block_iter_args[i]] = iter_args[i]
        for i in range(lb_int, ub_int, step_int):
            for cur_op in op.body.block.ops:
                if not isinstance(cur_op, transfer.NextLoopOp):
                    clone_op = cur_op.clone()
                    for idx in range(len(clone_op.operands)):
                        if cur_op.operands[idx] in value_map:
                            clone_op.operands[idx] = value_map[cur_op.operands[idx]]
                    if len(cur_op.results) != 0:
                        value_map[cur_op.results[0]] = clone_op.results[0]
                    rewriter.insert_op_before_matched_op(clone_op)
                    continue
                if isinstance(cur_op, transfer.NextLoopOp):
                    if i + step_int < ub_int:
                        new_value_map: dict[SSAValue, SSAValue] = {}
                        cur_ind = transfer.Constant(op.ub, i + step_int).result
                        new_value_map[indvar] = cur_ind
                        rewriter.insert_op_before_matched_op(cur_ind.owner)
                        for idx in range(len(block_iter_args)):
                            new_value_map[block_iter_args[idx]] = value_map[
                                cur_op.operands[idx]
                            ]
                        value_map = new_value_map
                    else:
                        make_res = [value_map[arg] for arg in cur_op.arguments]
                        """
                        make_op = transfer.MakeOp.create(
                            operands=make_res,
                            result_types=[
                                transfer.MakeOp.infer_result_type(
                                    [arg.type for arg in make_res]
                                )
                            ],
                        )
                        """
                        assert (
                            len(make_res) == 1
                            and "current we only support for one returned value from for"
                        )
                        rewriter.replace_matched_op([], [make_res[0]])
                        # rewriter.replace_matched_op(make_op)


@dataclass(frozen=True)
class UnrollTransferLoop(ModulePass):
    name = "unrollTransferLoop"

    width: int

    def apply(self, ctx: Context, op: ModuleOp):
        walker = PatternRewriteWalker(
            ConstRangeForOpPattern(self.width), walk_reverse=True
        )
        walker.rewrite_module(op)
