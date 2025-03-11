from abc import abstractmethod, ABC
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.ir import SSAValue, Attribute
from xdsl.dialects import builtin, arith


class HasInhabitant(ABC):
    """Return an inhabitant of the type."""

    @classmethod
    @abstractmethod
    def create_inhabitant(cls, rewriter: PatternRewriter) -> SSAValue:
        ...


def create_inhabitant(type: Attribute, rewriter: PatternRewriter) -> SSAValue | None:
    if isinstance(type, HasInhabitant):
        return type.create_inhabitant(rewriter)
    if isinstance(type, builtin.IntegerType):
        constant = arith.ConstantOp(builtin.IntegerAttr(0, type), type)
        rewriter.insert_op_before_matched_op(constant)
        return constant.result
