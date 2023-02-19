"""
Remove all pairs from the program.
This duplicate all functions that return a pair, into two functions.
It also duplicate arguments that are pairs, into two arguments.
"""

from typing import cast
from xdsl.dialects.builtin import ArrayAttr, FunctionType, ModuleOp
from xdsl.ir import Attribute, MLContext, Operation
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker, PatternRewriter, RewritePattern, op_type_rewrite_pattern

from dialects.smt_dialect import CallOp, DefineFunOp
from dialects.smt_utils_dialect import AnyPairType, FirstOp, PairOp, PairType, SecondOp
from passes.canonicalize_smt import FoldUtilsPattern


class RemovePairArgsFunction(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DefineFunOp, rewriter: PatternRewriter):
        block = op.body.blocks[0]
        i = 0
        while i < len(block.args):
            arg = block.args[i]
            if isinstance(arg.typ, PairType):
                typ = cast(AnyPairType, arg.typ)
                fst = rewriter.insert_block_argument(block, i + 1, typ.first)
                snd = rewriter.insert_block_argument(block, i + 2, typ.second)
                get_pair = PairOp.from_values(fst, snd)
                rewriter.insert_op_at_pos(get_pair, block, 0)
                arg.replace_by(get_pair.res)
                rewriter.erase_block_argument(arg)
            else:
                i += 1
        old_typ = op.ret.typ
        assert isinstance(old_typ, FunctionType)
        new_inputs = [arg.typ for arg in block.args]
        op.ret.typ = FunctionType.from_attrs(
            ArrayAttr[Attribute].from_list(new_inputs), old_typ.outputs)


class RemovePairArgsCall(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter):
        args = op.args
        i = 0
        while i < len(args):
            arg = args[i]
            if isinstance(arg.typ, PairType):
                fst = FirstOp.from_value(arg)
                snd = SecondOp.from_value(arg)
                rewriter.insert_op_before_matched_op([fst, snd])
                rewriter.replace_matched_op(
                    CallOp.create(result_types=[res.typ for res in op.results],
                                  operands=args[:i] + [fst.res, snd.res] +
                                  args[i + 1:]))
                return
            else:
                i += 1


def lower_pairs(ctx: MLContext, module: ModuleOp):
    # Remove pairs from function arguments.
    walker = PatternRewriteWalker(
        GreedyRewritePatternApplier([
            RemovePairArgsFunction(),
            RemovePairArgsCall(),
        ]))
    walker.rewrite_module(module)

    # Simplify pairs away
    walker = PatternRewriteWalker(FoldUtilsPattern())
    walker.rewrite_module(module)