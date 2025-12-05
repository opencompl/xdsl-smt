from __future__ import annotations

from typing import ClassVar, TypeVar, IO, overload

from xdsl.dialects.builtin import IntAttr, IntegerAttr, IntegerType

from xdsl.ir import (
    Attribute,
    Dialect,
    OpResult,
    SSAValue,
    VerifyException,
)
from xdsl.irdl import (
    attr_def,
    operand_def,
    result_def,
    Operand,
    irdl_op_definition,
    IRDLOperation,
    traits_def,
    VarConstraint,
    base,
    prop_def,
)
from xdsl import traits
from xdsl.traits import HasCanonicalizationPatternsTrait
from xdsl.pattern_rewriter import RewritePattern

from xdsl_smt.traits.smt_printer import SMTConversionCtx, SMTLibOp, SimpleSMTLibOp
from xdsl_smt.traits.effects import Pure
from xdsl_smt.dialects.smt_dialect import BoolType
from xdsl.dialects.smt import BitVectorType, BitVectorAttr


@irdl_op_definition
class ConstantOp(IRDLOperation, Pure, SMTLibOp):
    name = "smt.bv.constant"

    T: ClassVar = VarConstraint("T", base(BitVectorType))

    value = prop_def(BitVectorAttr.constr(T))
    result = result_def(T)

    assembly_format = "qualified($value) attr-dict"

    @property
    def res(self) -> OpResult[BitVectorType]:
        return self.result

    @overload
    def __init__(self, value: int | IntAttr, width: int | IntAttr) -> None:
        ...

    @overload
    def __init__(self, value: IntegerAttr[IntegerType] | BitVectorAttr) -> None:
        ...

    def __init__(
        self,
        value: int | IntAttr | IntegerAttr[IntegerType] | BitVectorAttr,
        width: int | IntAttr | None = None,
    ) -> None:
        attr: BitVectorAttr
        if isinstance(value, int | IntAttr):
            if not isinstance(width, int | IntAttr):
                raise ValueError("Expected width with an `int` value")
            attr = BitVectorAttr(value, BitVectorType(width))
        elif isinstance(value, BitVectorAttr):
            attr = value
        else:
            width = value.type.width.data
            value_int = (
                value.value.data + 2**width
                if value.value.data < 0
                else value.value.data
            )
            attr = BitVectorAttr(value_int, BitVectorType(width))
        super().__init__(result_types=[attr.get_type()], properties={"value": attr})

    @staticmethod
    def from_int_value(value: int, width: int) -> ConstantOp:
        bv_value = BitVectorAttr(value, BitVectorType(width))
        return ConstantOp.create(
            result_types=[bv_value.get_type()], properties={"value": bv_value}
        )

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx) -> None:
        print(
            f"(_ bv{self.value.value.data} {self.value.type.width.data})",
            file=stream,
            end="",
        )


_UOpT = TypeVar("_UOpT", bound="UnaryBVOp")


class UnaryBVOp(IRDLOperation, Pure):
    res: OpResult = result_def(BitVectorType)
    arg: Operand = operand_def(BitVectorType)

    def __init__(self, arg: SSAValue):
        super().__init__(result_types=[arg.type], operands=[arg])

    @classmethod
    def get(cls: type[_UOpT], arg: SSAValue) -> _UOpT:
        return cls.create(result_types=[arg.type], operands=[arg])

    def verify_(self):
        if not (self.res.type == self.arg.type):
            raise VerifyException("Operand and result must have the same type")


_BOpT = TypeVar("_BOpT", bound="BinaryBVOp")


class BinaryBVOp(IRDLOperation, Pure):
    res: OpResult = result_def(BitVectorType)
    lhs: Operand = operand_def(BitVectorType)
    rhs: Operand = operand_def(BitVectorType)

    def __init__(self, lhs: Operand, rhs: Operand, res: Attribute | None = None):
        if res is None:
            res = lhs.type
        super().__init__(result_types=[lhs.type], operands=[lhs, rhs])

    @classmethod
    def get(cls: type[_BOpT], lhs: SSAValue, rhs: SSAValue) -> _BOpT:
        return cls.create(result_types=[lhs.type], operands=[lhs, rhs])

    def verify_(self):
        if not (self.res.type == self.lhs.type == self.rhs.type):
            raise VerifyException("Operands must have same type")


################################################################################
#                          Basic Bitvector Arithmetic                          #
################################################################################


class AddCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            AddCanonicalizationPattern,
        )

        return (AddCanonicalizationPattern(),)


@irdl_op_definition
class AddOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.add"

    traits = traits_def(traits.Pure(), AddCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvadd"


class SubCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            SubCanonicalizationPattern,
        )

        return (SubCanonicalizationPattern(),)


@irdl_op_definition
class SubOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.sub"

    traits = traits_def(traits.Pure(), SubCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvsub"


@irdl_op_definition
class NegOp(UnaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.neg"

    def op_name(self) -> str:
        return "bvneg"


class MulCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            MulCanonicalizationPattern,
        )

        return (MulCanonicalizationPattern(),)


@irdl_op_definition
class MulOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.mul"

    traits = traits_def(traits.Pure(), MulCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvmul"


class URemCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import URemFold

        return (URemFold(),)


@irdl_op_definition
class URemOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.urem"

    traits = traits_def(traits.Pure(), URemCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvurem"


class SRemCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import SRemFold

        return (SRemFold(),)


@irdl_op_definition
class SRemOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.srem"

    traits = traits_def(traits.Pure(), SRemCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvsrem"


@irdl_op_definition
class SModOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.smod"

    def op_name(self) -> str:
        return "bvsmod"


@irdl_op_definition
class ShlOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.shl"

    def op_name(self) -> str:
        return "bvshl"


@irdl_op_definition
class LShrOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.lshr"

    def op_name(self) -> str:
        return "bvlshr"


@irdl_op_definition
class AShrOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.ashr"

    def op_name(self) -> str:
        return "bvashr"


class SDivCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import SDivFold

        return (SDivFold(),)


@irdl_op_definition
class SDivOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.sdiv"

    traits = traits_def(traits.Pure(), SDivCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvsdiv"


class UDivCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import UDivFold

        return (UDivFold(),)


@irdl_op_definition
class UDivOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.udiv"

    traits = traits_def(traits.Pure(), UDivCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvudiv"


################################################################################
#                                   Bitwise                                    #
################################################################################


class OrCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            OrCanonicalizationPattern,
        )

        return (OrCanonicalizationPattern(),)


@irdl_op_definition
class OrOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.or"

    traits = traits_def(traits.Pure(), OrCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvor"


class AndCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            AndCanonicalizationPattern,
        )

        return (AndCanonicalizationPattern(),)


@irdl_op_definition
class AndOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.and"

    traits = traits_def(traits.Pure(), AndCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvand"


@irdl_op_definition
class NotOp(UnaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.not"

    def op_name(self) -> str:
        return "bvnot"


class XorCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            XorCanonicalizationPattern,
        )

        return (XorCanonicalizationPattern(),)


@irdl_op_definition
class XorOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.xor"

    traits = traits_def(traits.Pure(), XorCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvxor"


@irdl_op_definition
class NAndOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.nand"

    def op_name(self) -> str:
        return "bvnand"


@irdl_op_definition
class NorOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.nor"

    def op_name(self) -> str:
        return "bvnor"


@irdl_op_definition
class XNorOp(BinaryBVOp, SimpleSMTLibOp):
    name = "smt.bv.xnor"

    def op_name(self) -> str:
        return "bvxnor"


################################################################################
#                                  Predicate                                   #
################################################################################

_BPOpT = TypeVar("_BPOpT", bound="BinaryPredBVOp")


class UnaryPredBVOp(IRDLOperation, Pure):
    res: OpResult = result_def(BoolType)
    operand: Operand = operand_def(BitVectorType)

    def __init__(self, operand: SSAValue):
        super().__init__(result_types=[BoolType()], operands=[operand])


class BinaryPredBVOp(IRDLOperation, Pure):
    res: OpResult = result_def(BoolType)
    lhs: Operand = operand_def(BitVectorType)
    rhs: Operand = operand_def(BitVectorType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(result_types=[BoolType()], operands=[lhs, rhs])

    @classmethod
    def get(cls: type[_BPOpT], lhs: SSAValue, rhs: SSAValue) -> _BPOpT:
        return cls.create(result_types=[BoolType()], operands=[lhs, rhs])

    def verify_(self):
        if not (self.lhs.type == self.rhs.type):
            raise VerifyException("Operands must have the same type")


class UleCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            UleCanonicalizationPattern,
        )

        return (UleCanonicalizationPattern(),)


@irdl_op_definition
class CmpOp(SimpleSMTLibOp, IRDLOperation, Pure):
    name = "smt.bv.cmp"

    res = result_def(BoolType)
    lhs = operand_def(BitVectorType)
    rhs = operand_def(BitVectorType)

    pred = prop_def(IntegerAttr[IntegerType])

    def __init__(self, pred: IntegerAttr[IntegerType], lhs: SSAValue, rhs: SSAValue):
        super().__init__(
            result_types=[BoolType()], operands=[lhs, rhs], properties={"pred": pred}
        )

    traits = traits_def(traits.Pure())

    def op_name(self) -> str:
        return {
            0: "bvslt",
            1: "bvsle",
            2: "bvsgt",
            3: "bvsge",
            4: "bvult",
            5: "bvule",
            6: "bvugt",
            7: "bvuge",
        }[self.pred.value.data]


@irdl_op_definition
class UleOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.ule"

    traits = traits_def(traits.Pure(), UleCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvule"


class UltCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            UltCanonicalizationPattern,
        )

        return (UltCanonicalizationPattern(),)


@irdl_op_definition
class UltOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.ult"

    traits = traits_def(traits.Pure(), UltCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvult"


class UgeCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            UgeCanonicalizationPattern,
        )

        return (UgeCanonicalizationPattern(),)


@irdl_op_definition
class UgeOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.uge"

    traits = traits_def(traits.Pure(), UgeCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvuge"


class UgtCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            UgtCanonicalizationPattern,
        )

        return (UgtCanonicalizationPattern(),)


@irdl_op_definition
class UgtOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.ugt"

    traits = traits_def(traits.Pure(), UgtCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvugt"


class SleCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            SleCanonicalizationPattern,
        )

        return (SleCanonicalizationPattern(),)


@irdl_op_definition
class SleOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.sle"

    traits = traits_def(traits.Pure(), SleCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvsle"


class SltCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            SltCanonicalizationPattern,
        )

        return (SltCanonicalizationPattern(),)


@irdl_op_definition
class SltOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.slt"

    traits = traits_def(traits.Pure(), SltCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvslt"


class SgeCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            SgeCanonicalizationPattern,
        )

        return (SgeCanonicalizationPattern(),)


@irdl_op_definition
class SgeOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.sge"

    traits = traits_def(traits.Pure(), SgeCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvsge"


class SgtCanonicalizationPatterns(HasCanonicalizationPatternsTrait):
    @classmethod
    def get_canonicalization_patterns(cls) -> tuple[RewritePattern, ...]:
        from xdsl_smt.passes.canonicalization_patterns.smt_bv import (
            SgtCanonicalizationPattern,
        )

        return (SgtCanonicalizationPattern(),)


@irdl_op_definition
class SgtOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.sgt"

    traits = traits_def(traits.Pure(), SgtCanonicalizationPatterns())

    def op_name(self) -> str:
        return "bvsgt"


@irdl_op_definition
class UmulNoOverflowOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.umul_noovfl"

    traits = traits_def(traits.Pure())

    def op_name(self) -> str:
        return "bvumul_noovfl"


@irdl_op_definition
class SmulNoOverflowOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.smul_noovfl"

    traits = traits_def(traits.Pure())

    def op_name(self) -> str:
        return "bvsmul_noovfl"


@irdl_op_definition
class SmulNoUnderflowOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.smul_noudfl"

    traits = traits_def(traits.Pure())

    def op_name(self) -> str:
        return "bvsmul_noudfl"


@irdl_op_definition
class NegOverflowOp(UnaryPredBVOp, SimpleSMTLibOp):
    """
    [[(bvnego s)]] := bv2int([[s]]) == -2^(m - 1)

    We define an extra rewriting pass to implement it because Z3 doesn't support it
    """

    name = "smt.bv.nego"

    traits = traits_def(traits.Pure())

    def op_name(self) -> str:
        return "bvnego"


@irdl_op_definition
class UaddOverflowOp(BinaryPredBVOp, SimpleSMTLibOp):
    """
    [[(bvuaddo s t)]] := (bv2nat([[s]]) + bv2nat([[t]])) >= 2^m

    We define an extra rewriting pass to implement it because Z3 doesn't support it
    """

    name = "smt.bv.uaddo"

    traits = traits_def(traits.Pure())

    def op_name(self) -> str:
        return "bvuaddo"


@irdl_op_definition
class SaddOverflowOp(BinaryPredBVOp, SimpleSMTLibOp):
    """
    [[(bvsaddo s t)]] := (bv2int([[s]]) + bv2int([[t]])) >= 2^(m - 1) or
                         (bv2int([[s]]) + bv2int([[t]])) < -2^(m - 1)

    We define an extra rewriting pass to implement it because Z3 doesn't support it
    """

    name = "smt.bv.saddo"

    traits = traits_def(traits.Pure())

    def op_name(self) -> str:
        return "bvsaddo"


@irdl_op_definition
class UsubOverflowOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.usubo"

    traits = traits_def(traits.Pure())

    def op_name(self) -> str:
        return "bvusubo"


@irdl_op_definition
class SsubOverflowOp(BinaryPredBVOp, SimpleSMTLibOp):
    name = "smt.bv.ssubo"

    traits = traits_def(traits.Pure())

    def op_name(self) -> str:
        return "bvssubo"


@irdl_op_definition
class UmulOverflowOp(BinaryPredBVOp, SimpleSMTLibOp):
    """
    [[(bvumulo s t)]] := (bv2nat([[s]]) * bv2nat([[t]])) >= 2^m

    We define an extra rewriting pass to implement it because Z3 doesn't support it
    """

    name = "smt.bv.umulo"

    traits = traits_def(traits.Pure())

    def op_name(self) -> str:
        return "bvumulo"


@irdl_op_definition
class SmulOverflowOp(BinaryPredBVOp, SimpleSMTLibOp):
    """
    [[(bvsmulo s t)]] := (bv2int([[s]]) * bv2int([[t]])) >= 2^(m - 1) or
                         (bv2int([[s]]) * bv2int([[t]])) < -2^(m - 1)

    We define an extra rewriting pass to implement it because Z3 doesn't support it
    """

    name = "smt.bv.smulo"

    traits = traits_def(traits.Pure())

    def op_name(self) -> str:
        return "bvsmulo"


################################################################################
#                                  Predicate                                   #
################################################################################


@irdl_op_definition
class ConcatOp(IRDLOperation, SimpleSMTLibOp):
    name = "smt.bv.concat"

    lhs: Operand = operand_def(BitVectorType)
    rhs: Operand = operand_def(BitVectorType)
    res: OpResult = result_def(BitVectorType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        assert isinstance(lhs.type, BitVectorType)
        assert isinstance(rhs.type, BitVectorType)
        width = lhs.type.width.data + rhs.type.width.data
        super().__init__(result_types=[BitVectorType(width)], operands=[lhs, rhs])

    def op_name(self) -> str:
        return "concat"


@irdl_op_definition
class ExtractOp(IRDLOperation, SMTLibOp):
    name = "smt.bv.extract"

    operand: Operand = operand_def(BitVectorType)
    res: OpResult = result_def(BitVectorType)

    start: IntAttr = attr_def(IntAttr)
    end: IntAttr = attr_def(IntAttr)

    def __init__(self, operand: SSAValue, end: int, start: int):
        super().__init__(
            result_types=[BitVectorType(end - start + 1)],
            operands=[operand],
            attributes={"start": IntAttr(start), "end": IntAttr(end)},
        )

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx) -> None:
        """Print the operation to an SMTLib representation."""
        print(f"((_ extract {self.end.data} {self.start.data}) ", file=stream, end="")
        ctx.print_expr_to_smtlib(self.operand, stream)
        print(")", file=stream, end="")


@irdl_op_definition
class RepeatOp(IRDLOperation, SMTLibOp):
    name = "smt.bv.repeat"

    operand: Operand = operand_def(BitVectorType)
    res: OpResult = result_def(BitVectorType)

    count: IntAttr = attr_def(IntAttr)

    def __init__(self, operand: SSAValue, count: int):
        assert isinstance(operand.type, BitVectorType)
        assert count >= 1
        super().__init__(
            result_types=[BitVectorType(operand.type.width.data * count)],
            operands=[operand],
            attributes={"count": IntAttr(count)},
        )

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx) -> None:
        """Print the operation to an SMTLib representation."""
        print(f"((_ repeat {self.count.data}) ", file=stream, end="")
        ctx.print_expr_to_smtlib(self.operand, stream)
        print(")", file=stream, end="")


@irdl_op_definition
class ZeroExtendOp(IRDLOperation, SMTLibOp):
    name = "smt.bv.zero_extend"

    operand: Operand = operand_def(BitVectorType)
    res: OpResult = result_def(BitVectorType)

    def __init__(self, operand: SSAValue, res_type: BitVectorType):
        assert isinstance(operand.type, BitVectorType)
        assert res_type.width.data > operand.type.width.data

        super().__init__(
            result_types=[res_type],
            operands=[operand],
        )

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx) -> None:
        """Print the operation to an SMTLib representation."""
        assert isinstance(self.res.type, BitVectorType)
        assert isinstance(self.operand.type, BitVectorType)
        print(
            f"((_ zero_extend {self.res.type.width.data - self.operand.type.width.data}) ",
            file=stream,
            end="",
        )
        ctx.print_expr_to_smtlib(self.operand, stream)
        print(")", file=stream, end="")


@irdl_op_definition
class SignExtendOp(IRDLOperation, SMTLibOp):
    name = "smt.bv.sign_extend"

    operand: Operand = operand_def(BitVectorType)
    res: OpResult = result_def(BitVectorType)

    def __init__(self, operand: SSAValue, res_type: BitVectorType):
        assert isinstance(operand.type, BitVectorType)
        assert res_type.width.data > operand.type.width.data

        super().__init__(
            result_types=[res_type],
            operands=[operand],
        )

    def print_expr_to_smtlib(self, stream: IO[str], ctx: SMTConversionCtx) -> None:
        """Print the operation to an SMTLib representation."""
        assert isinstance(self.res.type, BitVectorType)
        assert isinstance(self.operand.type, BitVectorType)
        print(
            f"((_ sign_extend {self.res.type.width.data - self.operand.type.width.data}) ",
            file=stream,
            end="",
        )
        ctx.print_expr_to_smtlib(self.operand, stream)
        print(")", file=stream, end="")


SMTBitVectorDialect = Dialect(
    "smt.bv",
    [
        ConstantOp,
        # Arithmetic
        NegOp,
        AddOp,
        SubOp,
        MulOp,
        URemOp,
        SRemOp,
        SModOp,
        ShlOp,
        LShrOp,
        AShrOp,
        UDivOp,
        SDivOp,
        # Bitwise
        NotOp,
        OrOp,
        XorOp,
        AndOp,
        NAndOp,
        NorOp,
        XNorOp,
        # Predicate
        CmpOp,
        UleOp,
        UltOp,
        UgeOp,
        UgtOp,
        SleOp,
        SltOp,
        SgeOp,
        SgtOp,
        # Overflow Predicate
        NegOverflowOp,
        UaddOverflowOp,
        SaddOverflowOp,
        UsubOverflowOp,
        SsubOverflowOp,
        UmulOverflowOp,
        SmulOverflowOp,
        UmulNoOverflowOp,
        SmulNoOverflowOp,
        SmulNoUnderflowOp,
        # Others
        ConcatOp,
        ExtractOp,
        RepeatOp,
        ZeroExtendOp,
        SignExtendOp,
    ],
    [BitVectorType, BitVectorAttr],
)
