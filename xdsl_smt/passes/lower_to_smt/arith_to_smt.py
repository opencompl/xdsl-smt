from typing import Callable

from xdsl.ir import (Operation, SSAValue)
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl.utils.hints import isa

from ...utils.rewrite_tools import (
    PatternRegistrar,
    SimpleRewritePattern,
    SimpleRewritePatternFactory
)
from ...dialects import smt_dialect as smt
from ...dialects import smt_bitvector_dialect as bv
from ...dialects.smt_bitvector_dialect import BitVectorType
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


@rewrite_pattern
class CmpiRewritePattern(SimpleRewritePattern):
    @staticmethod
    def cmpOp(predicate: IntegerAttr[IntegerType]) -> type[smt.BinaryTOp | bv.BinaryPredBVOp]:
        return [smt.EqOp, smt.DistinctOp,
                bv.SltOp, bv.SleOp, bv.SgtOp, bv.SgeOp,
                bv.UltOp, bv.UleOp, bv.UgtOp, bv.UgeOp][predicate.value.data]

    @staticmethod
    def rewrite(src: arith.Cmpi) -> Operation:
        return smt.IteOp(SSAValue.get(__class__.cmpOp(src.predicate)(src.lhs, src.rhs)),
                         SSAValue.get(bv.ConstantOp(1, 1)), SSAValue.get(bv.ConstantOp(0, 1)))


@rewrite_pattern
class SelectRewritePattern(SimpleRewritePattern):
    @staticmethod
    def rewrite(src: arith.Select) -> Operation:
        return smt.IteOp(SSAValue.get(smt.EqOp(src.condition, SSAValue.get(bv.ConstantOp(1, 1)))),
                         src.true_value, src.false_value)


class _ExtTruncOps:
    @staticmethod
    def width(val: SSAValue) -> int:
        """Get the bit width of an integer-type value"""

        ty = val.type
        assert isinstance(ty, IntegerType) or isinstance(ty, BitVectorType)
        bits = ty.width.data
        return bits

    @staticmethod
    def extract_sign_bit(val: SSAValue) -> Operation:
        """Create a SMT expression that extracts the sign bit from an integer-type value"""

        bits = __class__.width(val)
        return bv.ExtractOp(val, bits - 1, bits - 1)


class ExtTruncRewriteBase(SimpleRewritePattern, _ExtTruncOps):
    pass


@rewrite_pattern
class TrunciRewritePattern(ExtTruncRewriteBase):
    @staticmethod
    def rewrite(src: arith.Trunci) -> Operation:
        bits = __class__.width(src.out)
        return bv.ExtractOp(src.in_, bits - 1, 0)


@rewrite_pattern
class ExtuiRewritePattern(ExtTruncRewriteBase):
    @staticmethod
    def rewrite(src: arith.Extui) -> Operation:
        bits = __class__.width(src.out) - __class__.width(src.in_)
        extension = SSAValue.get(bv.ConstantOp(0, bits))
        return bv.ConcatOp(extension, src.in_)


@rewrite_pattern
class ExtsiRewritePattern(ExtTruncRewriteBase):
    @staticmethod
    def rewrite(src: arith.Extsi) -> Operation:
        bits = __class__.width(src.out) - __class__.width(src.in_)
        sign_bit = SSAValue.get(__class__.extract_sign_bit(src.in_))
        extension = SSAValue.get(smt.IteOp(
            SSAValue.get(smt.EqOp(sign_bit, SSAValue.get(bv.ConstantOp(0, 1)))),
            SSAValue.get(bv.ConstantOp(0, bits)),
            SSAValue.get(bv.ConstantOp(2 ** bits - 1, bits))))
        return bv.ConcatOp(extension, src.in_)


@rewrite_pattern
class AdduiCarryRewritePattern(RewritePattern, _ExtTruncOps):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, src: arith.AdduiCarry, rewriter: PatternRewriter) -> None:
        # Extended the bitvectors by 1 bit, do the addition, and extract the MSB as the carry bit

        _g = SSAValue.get

        bits = __class__.width(src.sum)
        zero = bv.ConstantOp(0, 1)
        lhs = bv.ConcatOp(_g(zero), _g(src.lhs))
        rhs = bv.ConcatOp(_g(zero), _g(src.rhs))
        full_sum = bv.AddOp(_g(lhs), _g(rhs))
        result = bv.ExtractOp(_g(full_sum), bits - 1, 0)
        overflow = __class__.extract_sign_bit(_g(full_sum))
        rewriter.replace_matched_op([zero, lhs, rhs, full_sum, result, overflow],
                                    [_g(result), _g(overflow)])
