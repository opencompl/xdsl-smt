from xdsl.ir import Attribute, SSAValue
from xdsl.parser import IntegerAttr, IntegerType, IndexType, i64
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.hints import isa
from xdsl_smt.semantics.semantics import (
    AttributeSemantics,
    TypeSemantics,
)

from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_utils_dialect as smt_utils


class IntegerTypeSemantics(TypeSemantics):
    """Convert an integer type to a bitvector integer with a poison flag."""

    def get_semantics(self, type: Attribute) -> Attribute:
        assert isinstance(type, IntegerType)
        return smt_utils.PairType(smt_bv.BitVectorType(type.width), smt.BoolType())


class IndexTypeSemantics(TypeSemantics):
    """
    Convert an index type to a bitvector integer with a poison flag.
    Index types are currently expected to be 64 bits wide integers.
    """

    def get_semantics(self, type: Attribute) -> Attribute:
        assert isinstance(type, IndexType)
        return smt_utils.PairType(smt_bv.BitVectorType(64), smt.BoolType())


class IntegerAttrSemantics(AttributeSemantics):
    def get_semantics(
        self, attribute: Attribute, rewriter: PatternRewriter
    ) -> SSAValue:
        if isa(attribute, IntegerAttr[IndexType]):
            attribute = IntegerAttr(attribute.value.data, i64)
        if not isa(attribute, IntegerAttr[IntegerType]):
            raise Exception(f"Cannot handle semantics of {attribute}")
        op = smt_bv.ConstantOp(attribute)
        rewriter.insert_op_before_matched_op(op)
        return op.res
