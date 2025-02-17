"""
Remove all pairs from the program.
This duplicate all functions that return a pair, into two functions.
It also duplicate arguments that are pairs, into two arguments.
"""

from typing import cast
from xdsl.dialects.builtin import ArrayAttr, FunctionType, ModuleOp, StringAttr
from xdsl.ir import Attribute
from xdsl.context import MLContext
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriteWalker,
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import Rewriter, InsertPoint
from xdsl.passes import ModulePass
from xdsl.utils.hints import isa

from ..dialects.smt_dialect import (
    CallOp,
    DeclareConstOp,
    DefineFunOp,
    ReturnOp,
    ForallOp,
    EqOp,
    DistinctOp,
    OrOp,
    AndOp,
)
from ..dialects.smt_utils_dialect import (
    AnyPairType,
    FirstOp,
    PairOp,
    PairType,
    SecondOp,
)
from xdsl_smt.passes.canonicalization_patterns.smt_utils import (
    FirstCanonicalizationPattern,
    SecondCanonicalizationPattern,
)
from .dead_code_elimination import DeadCodeElimination


class RemovePairArgsFunction(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DefineFunOp, rewriter: PatternRewriter):
        block = op.body.blocks[0]
        i = 0
        while i < len(block.args):
            arg = block.args[i]
            if isa(typ := arg.type, AnyPairType):
                fst = rewriter.insert_block_argument(block, i + 1, typ.first)
                snd = rewriter.insert_block_argument(block, i + 2, typ.second)
                get_pair = PairOp(fst, snd)
                rewriter.insert_op(get_pair, InsertPoint.at_start(block))
                arg.replace_by(get_pair.res)
                rewriter.erase_block_argument(arg)
            else:
                i += 1
        old_typ = op.ret.type
        assert isinstance(old_typ, FunctionType)
        new_inputs = [arg.type for arg in block.args]
        op.ret.type = FunctionType.from_attrs(
            ArrayAttr[Attribute](new_inputs), old_typ.outputs
        )


class RemovePairArgsCall(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter):
        args = op.args
        i = 0
        while i < len(args):
            arg = args[i]
            if isinstance(arg.type, PairType):
                fst = FirstOp(arg)
                snd = SecondOp(arg)
                rewriter.insert_op_before_matched_op([fst, snd])
                rewriter.replace_matched_op(
                    CallOp.get(
                        op.func,
                        list(args[:i]) + [fst.res, snd.res] + list(args[i + 1 :]),
                    )
                )
                return
            else:
                i += 1


def remove_pairs_from_function_return(module: ModuleOp):
    funcs = list[DefineFunOp]()
    for op in module.walk():
        if isinstance(op, DefineFunOp):
            funcs.append(op)

    while len(funcs) != 0:
        func = funcs[-1]
        funcs.pop()

        if len(func.func_type.outputs.data) != 1:
            if any(isinstance(t, PairType) for t in func.func_type.outputs.data):
                raise ValueError(
                    "lower-pairs do not handle functions with multiple results with pairs yet"
                )

        if isinstance((output := func.func_type.outputs.data[0]), PairType):
            output = cast(AnyPairType, output)

            # Create a new operation that will return the first element of the pair
            # by duplicating the current function.
            firstFunc = func.clone()
            parent_block = func.parent_block()
            assert parent_block is not None
            parent_block.insert_op_after(firstFunc, func)

            # Mutate the new function to return the first element of the pair.
            firstBlock = firstFunc.body.blocks[0]
            firstOp = FirstOp(firstFunc.return_values[0])
            firstBlockTerminator = firstBlock.last_op
            assert firstBlockTerminator is not None
            firstBlock.insert_op_before(firstOp, firstBlockTerminator)
            firstRet = ReturnOp(firstOp.res)
            Rewriter.replace_op(firstBlockTerminator, firstRet)
            firstFunc.ret.type = FunctionType.from_attrs(
                firstFunc.func_type.inputs,
                ArrayAttr[Attribute]([output.first]),
            )
            if firstFunc.fun_name:
                firstFunc.attributes["fun_name"] = StringAttr(
                    firstFunc.fun_name.data + "_first"
                )

            # Mutate the current function to return the second element of the pair.
            secondFunc = func
            secondBlock = secondFunc.body.blocks[0]
            secondBlockTerminator = secondBlock.last_op
            assert secondBlockTerminator is not None
            secondOp = SecondOp(secondFunc.return_values[0])
            secondBlock.insert_op_before(secondOp, secondBlockTerminator)
            secondRet = ReturnOp(secondOp.res)
            Rewriter.replace_op(secondBlockTerminator, secondRet)
            secondFunc.ret.type = FunctionType.from_attrs(
                secondFunc.func_type.inputs,
                ArrayAttr[Attribute]([output.second]),
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
                    result_types=[firstFunc.ret.type.outputs.data[0]],
                    operands=[firstFunc.ret] + list(call.args),
                )
                callSecond = CallOp.create(
                    result_types=[secondFunc.ret.type.outputs.data[0]],
                    operands=[secondFunc.ret] + list(call.args),
                )
                mergeCalls = PairOp(callFirst.res[0], callSecond.res[0])
                Rewriter.replace_op(call, [callFirst, callSecond, mergeCalls])


class LowerForallArgsPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, forall: ForallOp, rewriter: PatternRewriter):
        forall_block = forall.body.block

        for i, arg in enumerate(forall_block.args):
            if isa(arg.type, AnyPairType):
                second_arg = forall_block.insert_arg(arg.type.second, i)
                first_arg = forall_block.insert_arg(arg.type.first, i)
                pair_op = PairOp(first_arg, second_arg)
                assert forall_block.ops.first is not None
                forall_block.insert_op_before(pair_op, forall_block.ops.first)
                arg.replace_by(pair_op.res)
                forall_block.erase_arg(arg)
                rewriter.has_done_action = True
                break


class LowerDeclareConstPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DeclareConstOp, rewriter: PatternRewriter):
        if not isa((pair_type := op.res.type), AnyPairType):
            return

        name_hint = op.res.name_hint or "const"
        first = DeclareConstOp(pair_type.first)
        first.res.name_hint = name_hint + "_first"
        second = DeclareConstOp(pair_type.second)
        second.res.name_hint = name_hint + "_second"
        pair = PairOp(first.res, second.res)
        rewriter.replace_op(op, [first, second, pair])


class LowerDistinctPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: DistinctOp, rewriter: PatternRewriter):
        if not isa(op.lhs.type, AnyPairType):
            return

        """
        (distinct (pair lhsFirst lhsSecond) (pair rhsFirst rhsSecond))
        ->
        (or (distinct lhsFirst rhsFirst) (distinct lhsSecond rhsSecond))
        """
        lhsFirst = FirstOp(op.lhs)
        lhsSecond = SecondOp(op.lhs)
        rhsFirst = FirstOp(op.rhs)
        rhsSecond = SecondOp(op.rhs)
        firstDistinct = DistinctOp(lhsFirst.res, rhsFirst.res)
        secondDistinct = DistinctOp(lhsSecond.res, rhsSecond.res)
        orOp = OrOp(firstDistinct.res, secondDistinct.res)

        rewriter.replace_op(
            op,
            [
                lhsFirst,
                lhsSecond,
                rhsFirst,
                rhsSecond,
                firstDistinct,
                secondDistinct,
                orOp,
            ],
        )


class LowerEqPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: EqOp, rewriter: PatternRewriter):
        if not isa(op.lhs.type, AnyPairType):
            return

        """
        (= (pair lhsFirst lhsSecond) (pair rhsFirst rhsSecond))
        ->
        (and (= lhsFirst rhsFirst) (= lhsSecond rhsSecond))
        """
        lhsFirst = FirstOp(op.lhs)
        lhsSecond = SecondOp(op.lhs)
        rhsFirst = FirstOp(op.rhs)
        rhsSecond = SecondOp(op.rhs)
        firstEq = EqOp(lhsFirst.res, rhsFirst.res)
        secondEq = EqOp(lhsSecond.res, rhsSecond.res)
        andOp = AndOp(firstEq.res, secondEq.res)

        rewriter.replace_op(
            op,
            [
                lhsFirst,
                lhsSecond,
                rhsFirst,
                rhsSecond,
                firstEq,
                secondEq,
                andOp,
            ],
        )


class LowerPairs(ModulePass):
    name = "lower-pairs"

    def apply(self, ctx: MLContext, op: ModuleOp):
        # Remove pairs from function arguments.
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerForallArgsPattern(),
                    RemovePairArgsFunction(),
                    RemovePairArgsCall(),
                    LowerDeclareConstPattern(),
                    LowerEqPattern(),
                    LowerDistinctPattern(),
                ]
            )
        )
        walker.rewrite_module(op)

        # Remove pairs from function return.
        remove_pairs_from_function_return(op)

        # Simplify pairs away
        walker = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [FirstCanonicalizationPattern(), SecondCanonicalizationPattern()]
            )
        )
        walker.rewrite_module(op)
        # Apply DCE pass
        DeadCodeElimination().apply(ctx, op)
