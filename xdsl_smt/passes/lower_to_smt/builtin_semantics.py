from xdsl.ir import SSAValue
from xdsl.parser import AnyIntegerAttr, IntegerAttr, IntegerType
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.hints import isa
from xdsl_smt.passes.lower_to_smt.semantics import (
    AttributeSemantics,
)

from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv


class IntegerAttrSemantics(AttributeSemantics[AnyIntegerAttr]):
    def get_semantics(
        self, attribute: AnyIntegerAttr, rewriter: PatternRewriter
    ) -> SSAValue:
        if not isa(attribute, IntegerAttr[IntegerType]):
            raise Exception("Cannot handle semantics of IntegerAttr[IntegerType]")
        op = smt_bv.ConstantOp(attribute)
        rewriter.insert_op_before_matched_op(op)
        return op.res
