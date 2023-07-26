from typing import Callable

from xdsl.ir import (Operation, SSAValue)
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.utils.hints import isa

from ...utils.rewrite_tools import (PatternRegistrar, SimpleRewritePatternFactory)
from ...dialects import smt_dialect as smt
from ...dialects import smt_bitvector_dialect as bv
from ...dialects import arith_dialect as arith


rewrite_pattern: PatternRegistrar = PatternRegistrar()
arith_to_smt_patterns: list[RewritePattern] = rewrite_pattern.registry

_rewrite_factory = SimpleRewritePatternFactory(rewrite_pattern, globals())


@rewrite_pattern
class IntegerConstantRewritePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter):
        if not isa(op.value, IntegerAttr[IntegerType]):
            raise Exception("Cannot convert constant of type that are not integer type")
        smt_op = bv.ConstantOp(op.value)
        rewriter.replace_matched_op(smt_op)


for op in [
        # Arithmetic
        'Add', 'Sub', 'Mul',
        # Bitwise
        'And', 'Or', 'Xor', 'Shl'
        ]:
    srcOp = arith.__dict__[op + 'i']
    tgtOp = bv.__dict__[op + 'Op']
    _rewrite_factory.make_binop(srcOp, tgtOp)

for srcOp, tgtOp in [
        # Arithmetic
        (arith.Divsi, bv.SDivOp),
        (arith.Divui, bv.UDivOp),
        (arith.Remsi, bv.SRemOp),
        (arith.Remui, bv.URemOp),
        # Bitwise
        (arith.Shrsi, bv.AShrOp),
        (arith.Shrui, bv.LShrOp),
        ]:
    _rewrite_factory.make_binop(srcOp, tgtOp)

for srcOp, cmpOp in [
        (arith.Minsi, bv.SleOp),
        (arith.Minui, bv.UleOp),
        (arith.Maxsi, bv.SgeOp),
        (arith.Maxui, bv.UgeOp),
        ]:

    # Python lambdas capture by reference. Make sure we copy `cmpOp` by value.
    def mk_rewrite(cmpOp: type[Operation]) -> Callable[[Operation], smt.IteOp]:
        def rewrite(src: srcOp) -> smt.IteOp: # type: ignore
            return smt.IteOp(SSAValue.get(cmpOp(src.lhs, src.rhs)), src.lhs, src.rhs) # type: ignore
        return rewrite # type: ignore

    _rewrite_factory.make_simple(srcOp, mk_rewrite(cmpOp))
