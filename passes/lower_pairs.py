"""
Remove all pairs from the program.
This duplicate all functions that return a pair, into two functions.
It also duplicate arguments that are pairs, into two arguments.
"""

from typing import cast
from xdsl.dialects.builtin import ArrayAttr, FunctionType, ModuleOp, StringAttr
from xdsl.ir import Attribute, MLContext
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import Rewriter
from xdsl.passes import ModulePass

from dialects.smt_dialect import CallOp, DefineFunOp, ReturnOp
from dialects.smt_utils_dialect import AnyPairType, FirstOp, PairOp, PairType, SecondOp
from passes.canonicalize_smt import FoldUtilsPattern
from passes.dead_code_elimination import DeadCodeElimination


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
            ArrayAttr[Attribute].from_list(new_inputs), old_typ.outputs
        )


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
                    CallOp.create(
                        result_types=[res.typ for res in op.results],
                        operands=args[:i] + [fst.res, snd.res] + args[i + 1 :],
                    )
                )
                return
            else:
                i += 1


def remove_pairs_from_function_return(module: ModuleOp):
    funcs = list[DefineFunOp]()
    module.walk(lambda op: funcs.append(op) if isinstance(op, DefineFunOp) else None)

    while len(funcs) != 0:
        func = funcs[-1]
        funcs.pop()

        if isinstance((output := func.func_type.outputs.data[0]), PairType):
            output = cast(AnyPairType, output)

            # Create a new operation that will return the first element of the pair
            # by duplicating the current function.
            firstFunc = func.clone()
            parent_block = func.parent_block()
            assert parent_block is not None
            parent_block.insert_op(firstFunc, parent_block.get_operation_index(func))

            # Mutate the new function to return the first element of the pair.
            firstBlock = firstFunc.body.blocks[0]
            firstOp = FirstOp.from_value(firstFunc.return_val)
            firstBlock.insert_op(firstOp, len(firstBlock.ops) - 1)
            firstRet = ReturnOp.from_ret_value(firstOp.res)
            Rewriter.replace_op(firstBlock.ops[-1], firstRet)
            firstFunc.ret.typ = FunctionType.from_attrs(
                firstFunc.func_type.inputs,
                ArrayAttr[Attribute].from_list([output.first]),
            )
            if firstFunc.fun_name:
                firstFunc.attributes["fun_name"] = StringAttr(
                    firstFunc.fun_name.data + "_first"
                )

            # Mutate the current function to return the second element of the pair.
            secondFunc = func
            secondBlock = secondFunc.body.blocks[0]
            secondOp = SecondOp.from_value(secondFunc.return_val)
            secondBlock.insert_op(secondOp, len(secondBlock.ops) - 1)
            secondRet = ReturnOp.from_ret_value(secondOp.res)
            Rewriter.replace_op(secondBlock.ops[-1], secondRet)
            secondFunc.ret.typ = FunctionType.from_attrs(
                secondFunc.func_type.inputs,
                ArrayAttr[Attribute].from_list([output.second]),
            )
            if secondFunc.fun_name:
                secondFunc.attributes["fun_name"] = StringAttr(
                    secondFunc.fun_name.data + "_second"
                )

            funcs.append(firstFunc)
            funcs.append(secondFunc)

            # Replace all calls of this function by calls to the new functions.
            for use in set(func.ret.uses):
                call = use.operation
                assert isinstance(call, CallOp)
                callFirst = CallOp.create(
                    result_types=[firstFunc.ret.typ.outputs.data[0]],
                    operands=[firstFunc.ret] + list(call.args),
                )
                callSecond = CallOp.create(
                    result_types=[secondFunc.ret.typ.outputs.data[0]],
                    operands=[secondFunc.ret] + list(call.args),
                )
                mergeCalls = PairOp.from_values(callFirst.res, callSecond.res)
                Rewriter.replace_op(call, [callFirst, callSecond, mergeCalls])


class LowerPairs(ModulePass):
    name = "lower-pairs"

    def apply(self, ctx: MLContext, op: ModuleOp):
        # Remove pairs from function arguments.
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    RemovePairArgsFunction(),
                    RemovePairArgsCall(),
                ]
            )
        )
        walker.rewrite_module(op)

        # Remove pairs from function return.
        remove_pairs_from_function_return(op)

        # Simplify pairs away
        walker = PatternRewriteWalker(FoldUtilsPattern())
        walker.rewrite_module(op)

        # Apply DCE pass
        DeadCodeElimination().apply(ctx, op)
