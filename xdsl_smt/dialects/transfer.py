from __future__ import annotations
from abc import ABC

from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    i1,
)
from typing import ClassVar, Mapping, Sequence

from xdsl.ir import (
    ParametrizedAttribute,
    Dialect,
    TypeAttribute,
    OpResult,
    Region,
    Attribute,
    SSAValue,
)
from xdsl.utils.hints import isa

from xdsl.irdl import (
    attr_def,
    operand_def,
    var_operand_def,
    result_def,
    Operand,
    region_def,
    VarOperand,
    irdl_attr_definition,
    irdl_op_definition,
    param_def,
    IRDLOperation,
    traits_def,
    lazy_traits_def,
    AnyAttr,
    VarConstraint,
    irdl_to_attr_constraint,
)
from xdsl.utils.exceptions import VerifyException

from ..traits.infer_type import InferResultTypeInterface

from xdsl.traits import (
    HasParent,
    IsTerminator,
    SingleBlockImplicitTerminator,
)


@irdl_attr_definition
class TransIntegerType(ParametrizedAttribute, TypeAttribute):
    name = "transfer.integer"


@irdl_op_definition
class AddPoisonOp(IRDLOperation):
    name = "transfer.add_poison"
    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(TransIntegerType | IntegerType)
    )

    op: Operand = operand_def(T)
    result: OpResult = result_def(T)


@irdl_op_definition
class RemovePoisonOp(IRDLOperation):
    name = "transfer.remove_poison"
    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(TransIntegerType | IntegerType)
    )

    op: Operand = operand_def(T)
    result: OpResult = result_def(T)


@irdl_op_definition
class Constant(IRDLOperation, InferResultTypeInterface):
    name = "transfer.constant"

    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(TransIntegerType | IntegerType)
    )

    op: Operand = operand_def(T)
    result: OpResult = result_def(T)
    value: IntegerAttr[IndexType] = attr_def(IntegerAttr[IndexType])

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [op]:
                return [op]
            case _:
                raise VerifyException("Constant operation expects exactly one operand")

    def __init__(
        self,
        op: Operand,
        value: int,
    ):
        super().__init__(
            operands=[op],
            result_types=[op.type],
            attributes={"value": IntegerAttr(value, IndexType())},
        )


class UnaryOp(IRDLOperation, InferResultTypeInterface, ABC):
    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(TransIntegerType | IntegerType)
    )

    op: Operand = operand_def(T)
    result: OpResult = result_def(T)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [op]:
                return [op]
            case _:
                raise VerifyException("Unary operation expects exactly one operand")

    def __init__(
        self,
        op: SSAValue,
    ):
        super().__init__(
            operands=[op],
            result_types=[op.type],
        )


@irdl_op_definition
class NegOp(UnaryOp):
    name = "transfer.neg"


@irdl_op_definition
class ReverseBitsOp(UnaryOp):
    name = "transfer.reverse_bits"


class BinOp(IRDLOperation, InferResultTypeInterface, ABC):
    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(TransIntegerType | IntegerType)
    )

    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(T)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [lhs, _]:
                return [lhs]
            case _:
                raise VerifyException("Bin operation expects exactly two operands")

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[lhs.type],
        )


class PredicateOp(IRDLOperation, InferResultTypeInterface, ABC):
    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(TransIntegerType | IntegerType)
    )

    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(i1)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [_, _]:
                return [i1]
            case _:
                raise VerifyException("Bin operation expects exactly two operands")

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attributes: dict[str, Attribute] = {},
    ):
        super().__init__(operands=[lhs, rhs], result_types=[i1], attributes=attributes)


@irdl_op_definition
class AddOp(BinOp):
    name = "transfer.add"


@irdl_op_definition
class SubOp(BinOp):
    name = "transfer.sub"


@irdl_op_definition
class MulOp(BinOp):
    name = "transfer.mul"


@irdl_op_definition
class SDivOp(BinOp):
    name = "transfer.sdiv"


@irdl_op_definition
class UDivOp(BinOp):
    name = "transfer.udiv"


@irdl_op_definition
class SRemOp(BinOp):
    name = "transfer.srem"


@irdl_op_definition
class URemOp(BinOp):
    name = "transfer.urem"


@irdl_op_definition
class ShlOp(BinOp):
    name = "transfer.shl"


@irdl_op_definition
class AShrOp(BinOp):
    name = "transfer.ashr"


@irdl_op_definition
class LShrOp(BinOp):
    name = "transfer.lshr"


@irdl_op_definition
class UMulOverflowOp(PredicateOp):
    name = "transfer.umul_overflow"


@irdl_op_definition
class SMulOverflowOp(PredicateOp):
    name = "transfer.smul_overflow"


@irdl_op_definition
class UShlOverflowOp(PredicateOp):
    name = "transfer.ushl_overflow"


@irdl_op_definition
class SShlOverflowOp(PredicateOp):
    name = "transfer.sshl_overflow"


@irdl_op_definition
class UAddOverflowOp(PredicateOp):
    name = "transfer.uadd_overflow"


@irdl_op_definition
class SAddOverflowOp(PredicateOp):
    name = "transfer.sadd_overflow"


@irdl_op_definition
class USubOverflowOp(PredicateOp):
    name = "transfer.usub_overflow"


@irdl_op_definition
class SSubOverflowOp(PredicateOp):
    name = "transfer.ssub_overflow"


@irdl_op_definition
class AndOp(BinOp):
    name = "transfer.and"


@irdl_op_definition
class OrOp(BinOp):
    name = "transfer.or"


@irdl_op_definition
class XorOp(BinOp):
    name = "transfer.xor"


@irdl_op_definition
class IntersectsOp(BinOp):
    name = "transfer.intersects"


@irdl_op_definition
class GetBitWidthOp(UnaryOp):
    name = "transfer.get_bit_width"


@irdl_op_definition
class CountLZeroOp(UnaryOp):
    name = "transfer.countl_zero"


@irdl_op_definition
class CountRZeroOp(UnaryOp):
    name = "transfer.countr_zero"


@irdl_op_definition
class CountLOneOp(UnaryOp):
    name = "transfer.countl_one"


@irdl_op_definition
class CountROneOp(UnaryOp):
    name = "transfer.countr_one"


@irdl_op_definition
class PopCountOp(UnaryOp):
    name = "transfer.popcount"


@irdl_op_definition
class SMinOp(BinOp):
    name = "transfer.smin"


@irdl_op_definition
class SMaxOp(BinOp):
    name = "transfer.smax"


@irdl_op_definition
class UMinOp(BinOp):
    name = "transfer.umin"


@irdl_op_definition
class UMaxOp(BinOp):
    name = "transfer.umax"


@irdl_op_definition
class ConcatOp(BinOp):
    name = "transfer.concat"


@irdl_op_definition
class RepeatOp(BinOp):
    name = "transfer.repeat"


@irdl_op_definition
class ExtractOp(IRDLOperation):
    name = "transfer.extract"
    # the extracted bits [bitPosition,bitPosition+numBits].
    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(TransIntegerType | IntegerType)
    )

    val: Operand = operand_def(T)
    numBits: Operand = operand_def(T)
    bitPosition: Operand = operand_def(T)
    result: OpResult = result_def(T)

    def __init__(
        self,
        val: SSAValue,
        numBits: SSAValue,
        bitPosition: SSAValue,
    ):
        super().__init__(
            operands=[val, numBits, bitPosition],
            result_types=[val.type],
        )


@irdl_op_definition
class GetLowBitsOp(BinOp):
    """
    GetLowBitsOp(arg, numBits):
    Get low numBits
    """

    name = "transfer.get_low_bits"


@irdl_op_definition
class GetHighBitsOp(BinOp):
    """
    GetHighBitsOp(arg, numBits):
    Get high numBits
    """

    name = "transfer.get_high_bits"


@irdl_op_definition
class SetHighBitsOp(BinOp):
    """
    SetHighBitsOp(arg, numBits):
    Set high numBits to 1
    """

    name = "transfer.set_high_bits"


@irdl_op_definition
class SetLowBitsOp(BinOp):
    """
    SetLowBitsOp(arg, numBits):
    Set low numBits to 1
    """

    name = "transfer.set_low_bits"


@irdl_op_definition
class ClearHighBitsOp(BinOp):
    """
    ClearHighBitsOp(arg, numBits):
    Set top numBits to 0
    """

    name = "transfer.clear_high_bits"


@irdl_op_definition
class ClearLowBitsOp(BinOp):
    """
    ClearLowBitsOp(arg, numBits):
    Set low numBits to 0
    """

    name = "transfer.clear_low_bits"


@irdl_op_definition
class SetSignBitOp(UnaryOp):
    name = "transfer.set_sign_bit"


@irdl_op_definition
class ClearSignBitOp(UnaryOp):
    name = "transfer.clear_sign_bit"


class UnaryPredicateOp(IRDLOperation):
    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(TransIntegerType | IntegerType)
    )

    val: Operand = operand_def(T)
    result: OpResult = result_def(i1)

    def __init__(
        self,
        val: SSAValue,
    ):
        super().__init__(
            operands=[val],
            result_types=[i1],
        )


@irdl_op_definition
class IsPowerOf2Op(UnaryPredicateOp):
    name = "transfer.is_power_of_2"


@irdl_op_definition
class IsAllOnesOp(UnaryPredicateOp):
    name = "transfer.is_all_ones"


@irdl_op_definition
class IsNegativeOp(UnaryPredicateOp):
    name = "transfer.is_negative"


@irdl_op_definition
class CmpOp(PredicateOp):
    name = "transfer.cmp"

    predicate: IntegerAttr[IndexType] = attr_def(IntegerAttr[IndexType])

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        arg: int | str,
    ):
        if isinstance(arg, str):
            cmp_comparison_operations = {
                "eq": 0,
                "ne": 1,
                "slt": 2,
                "sle": 3,
                "sgt": 4,
                "sge": 5,
                "ult": 6,
                "ule": 7,
                "ugt": 8,
                "uge": 9,
            }
            assert arg in cmp_comparison_operations
            pred = cmp_comparison_operations[arg]
        else:
            pred = arg
        assert pred >= 0 and pred <= 9

        super().__init__(
            lhs,
            rhs,
            {"predicate": IntegerAttr.from_index_int_value(pred)},
        )


@irdl_attr_definition
class AbstractValueType(ParametrizedAttribute, TypeAttribute):
    name = "transfer.abs_value"
    fields: ArrayAttr[Attribute] = param_def()

    def get_num_fields(self) -> int:
        return len(self.fields.data)

    def get_fields(self):
        return [i for i in self.fields.data]

    def __init__(self, shape: list[Attribute] | ArrayAttr[Attribute]) -> None:
        if isinstance(shape, list):
            shape = ArrayAttr(shape)
        super().__init__(shape)


@irdl_attr_definition
class TupleType(ParametrizedAttribute, TypeAttribute):
    name = "transfer.tuple"
    fields: ArrayAttr[Attribute] = param_def()

    def get_num_fields(self) -> int:
        return len(self.fields.data)

    def get_fields(self):
        return [i for i in self.fields.data]

    def __init__(self, shape: list[Attribute] | ArrayAttr[Attribute]) -> None:
        if isinstance(shape, list):
            shape = ArrayAttr(shape)
        super().__init__(shape)


@irdl_op_definition
class GetOp(IRDLOperation, InferResultTypeInterface):
    name = "transfer.get"

    abs_val: Operand = operand_def(AbstractValueType)
    index: IntegerAttr[IndexType] = attr_def(IntegerAttr[IndexType])
    result: OpResult = result_def(Attribute)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        if len(operand_types) != 1 or not isinstance(
            operand_types[0], AbstractValueType
        ):
            raise VerifyException("Get operation expects exactly one abs_value operand")
        if "index" not in attributes:
            raise VerifyException("Get operation expects an index attribute")
        if not isa(attributes["index"], IntegerAttr[IndexType]):
            raise VerifyException("Get operation expects an integer index attribute")
        if attributes["index"].value.data >= operand_types[0].get_num_fields():
            raise VerifyException("'index' attribute is out of range")
        return [operand_types[0].get_fields()[attributes["index"].value.data]]

    def verify_(self) -> None:
        if self.infer_result_type(
            [operand.type for operand in self.operands], self.attributes
        ) != [self.result.type]:
            raise VerifyException("The result type doesn't match the inferred type")

    def __init__(
        self,
        op: Operand,
        index: int,
    ):
        assert isinstance(op.type, AbstractValueType) or isinstance(op.type, TupleType)
        ith_type = op.type.get_fields()[index]
        super().__init__(
            operands=[op],
            result_types=[ith_type],
            attributes={"index": IntegerAttr(index, IndexType())},
        )


@irdl_op_definition
class MakeOp(IRDLOperation, InferResultTypeInterface):
    name = "transfer.make"

    arguments: VarOperand = var_operand_def(Attribute)
    result: OpResult = result_def(AbstractValueType)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        return (AbstractValueType(list(operand_types)),)

    def verify_(self) -> None:
        assert isinstance(self.result.type, AbstractValueType)
        if len(self.operands) != self.result.type.get_num_fields():
            raise VerifyException(
                "The number of given arguments doesn't match the abstract value"
            )
        if self.result.type.get_fields() != [arg.type for arg in self.arguments]:
            raise VerifyException("The required field doesn't match the result type")

    def __init__(
        self,
        args: Sequence[SSAValue],
    ):
        arg_types = [arg.type for arg in args]
        result_type = AbstractValueType(arg_types)
        super().__init__(
            operands=[args],
            result_types=[result_type],
        )


@irdl_op_definition
class SelectOp(IRDLOperation):
    """
    Select between two values based on a condition.
    """

    name = "transfer.select"

    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(TransIntegerType | IntegerType)
    )

    cond: Operand = operand_def(IntegerType(1))
    true_value: Operand = operand_def(T)
    false_value: Operand = operand_def(T)
    result: OpResult = result_def(T)

    def __init__(
        self,
        cond: SSAValue,
        true_value: SSAValue,
        false_value: SSAValue,
    ):
        super().__init__(
            operands=[cond, true_value, false_value],
            result_types=[true_value.type],
        )


@irdl_op_definition
class NextLoopOp(IRDLOperation):
    name = "transfer.next_loop"
    arguments: VarOperand = var_operand_def(AnyAttr())
    traits = lazy_traits_def(lambda: (IsTerminator(), HasParent(ConstRangeForOp)))


@irdl_op_definition
class ConstRangeForOp(IRDLOperation):
    name = "transfer.const_range_for"

    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(TransIntegerType | IntegerType)
    )

    lb: Operand = operand_def(T)
    ub: Operand = operand_def(T)
    step: Operand = operand_def(T)

    iter_args: VarOperand = var_operand_def(AnyAttr())

    res: OpResult = result_def(AbstractValueType)

    body: Region = region_def("single_block")

    traits = traits_def(SingleBlockImplicitTerminator(NextLoopOp))

    def verify_(self):
        # body block verification
        if not self.body.block.args:
            raise VerifyException(
                "Body block must have induction var as first block arg"
            )

        indvar, *block_iter_args = self.body.block.args
        block_iter_args_num = len(block_iter_args)
        iter_args = self.iter_args
        iter_args_num = len(self.iter_args)

        for opnd in (self.lb, self.ub, self.step):
            if opnd.type != indvar.type:
                raise VerifyException(
                    "Expected induction var to be same type as bounds and step"
                )
            assert (
                isinstance(opnd.type, Constant)
                and "Const for requires bounds has to be constant"
            )
        if iter_args_num + 1 != block_iter_args_num + 1:
            raise VerifyException(
                f"Expected {iter_args_num + 1} args, but got {block_iter_args_num + 1}. "
                "Body block must have induction and loop-carried variables as args."
            )
        for i, arg in enumerate(iter_args):
            if block_iter_args[i].type != arg.type:
                raise VerifyException(
                    f"Block arg #{i + 1} expected to be {arg.type}, but got {block_iter_args[i].type}. "
                    "Block args after the induction variable must match the loop-carried variables."
                )
        if isinstance(last_op := self.body.block.last_op, NextLoopOp):
            return_val = last_op.arguments
            assert (
                isinstance(return_val, AbstractValueType)
                and "Returned from loop has to be an abstract val"
            )


@irdl_op_definition
class GetAllOnesOp(UnaryOp):
    """
    A special case of constant, return a bit vector with all bits set
    """

    name = "transfer.get_all_ones"

    T: ClassVar = VarConstraint(
        "T", irdl_to_attr_constraint(TransIntegerType | IntegerType)
    )

    op: Operand = operand_def(T)
    result: OpResult = result_def(T)

    @staticmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        match operand_types:
            case [op]:
                return [op]
            case _:
                raise VerifyException("Constant operation expects exactly one operand")


@irdl_op_definition
class GetSignedMaxValueOp(UnaryOp):
    """
    A special case of constant, return singed max value
    """

    name = "transfer.get_signed_max_value"


@irdl_op_definition
class GetSignedMinValueOp(UnaryOp):
    """
    A special case of constant, return signed min value
    """

    name = "transfer.get_signed_min_value"


Transfer = Dialect(
    "transfer",
    [
        Constant,
        CmpOp,
        AndOp,
        OrOp,
        XorOp,
        AddOp,
        SubOp,
        GetOp,
        MakeOp,
        NegOp,
        MulOp,
        SDivOp,
        UDivOp,
        SRemOp,
        URemOp,
        ShlOp,
        AShrOp,
        LShrOp,
        CountLOneOp,
        CountLZeroOp,
        CountROneOp,
        CountRZeroOp,
        PopCountOp,
        SetHighBitsOp,
        SetLowBitsOp,
        SetSignBitOp,
        ClearSignBitOp,
        GetLowBitsOp,
        GetHighBitsOp,
        ClearLowBitsOp,
        ClearHighBitsOp,
        GetBitWidthOp,
        SMinOp,
        SMaxOp,
        UMaxOp,
        UMinOp,
        UMulOverflowOp,
        SMulOverflowOp,
        UAddOverflowOp,
        SAddOverflowOp,
        USubOverflowOp,
        SSubOverflowOp,
        UShlOverflowOp,
        SShlOverflowOp,
        SelectOp,
        IsPowerOf2Op,
        IsAllOnesOp,
        IsNegativeOp,
        ConcatOp,
        RepeatOp,
        ExtractOp,
        ConstRangeForOp,
        NextLoopOp,
        GetAllOnesOp,
        GetSignedMaxValueOp,
        GetSignedMinValueOp,
        IntersectsOp,
        AddPoisonOp,
        RemovePoisonOp,
        ReverseBitsOp,
    ],
    [TransIntegerType, AbstractValueType, TupleType],
)
