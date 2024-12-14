from __future__ import annotations
from abc import ABC

from xdsl.dialects.builtin import (
    ArrayAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    i1,
)
from typing import Annotated, Mapping, Sequence

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
    ConstraintVar,
    attr_def,
    operand_def,
    var_operand_def,
    result_def,
    Operand,
    region_def,
    VarOperand,
    irdl_attr_definition,
    irdl_op_definition,
    ParameterDef,
    IRDLOperation,
    traits_def,
    AnyAttr,
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
    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    op: Operand = operand_def(T)
    result: OpResult = result_def(T)


@irdl_op_definition
class RemovePoisonOp(IRDLOperation):
    name = "transfer.remove_poison"
    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    op: Operand = operand_def(T)
    result: OpResult = result_def(T)


@irdl_op_definition
class Constant(IRDLOperation, InferResultTypeInterface):
    name = "transfer.constant"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

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
            properties={"value": IntegerAttr(value, IndexType())},
        )


class UnaryOp(IRDLOperation, InferResultTypeInterface, ABC):
    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

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
    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

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
    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

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
        properties: dict[str, Attribute] = {},
    ):
        super().__init__(operands=[lhs, rhs], result_types=[i1], properties=properties)


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
class ShlOverflowOp(PredicateOp):
    name = "transfer.shl_overflow"


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
class setAllBitsOp(UnaryOp):
    name = "transfer.set_all_bits"


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
    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

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
    name = "transfer.get_low_bits"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    val: Operand = operand_def(T)
    low_bits: Operand = operand_def(T)
    result: OpResult = result_def(T)


@irdl_op_definition
class SetHighBitsOp(BinOp):
    name = "transfer.set_high_bits"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    val: Operand = operand_def(T)
    high_bits: Operand = operand_def(T)
    result: OpResult = result_def(T)


@irdl_op_definition
class SetLowBitsOp(BinOp):
    name = "transfer.set_low_bits"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    val: Operand = operand_def(T)
    low_bits: Operand = operand_def(T)
    result: OpResult = result_def(T)


@irdl_op_definition
class SetSignBitOp(BinOp):
    name = "transfer.set_sign_bit"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    val: Operand = operand_def(T)
    sign_bit: Operand = operand_def(T)
    result: OpResult = result_def(T)


@irdl_op_definition
class IsPowerOf2Op(IRDLOperation):
    name = "transfer.is_power_of_2"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

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
class IsAllOnesOp(IRDLOperation):
    name = "transfer.is_all_ones"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

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
    fields: ParameterDef[ArrayAttr[Attribute]]

    def get_num_fields(self) -> int:
        return len(self.fields.data)

    def get_fields(self):
        return [i for i in self.fields.data]

    def __init__(self, shape: list[Attribute] | ArrayAttr[Attribute]) -> None:
        if isinstance(shape, list):
            shape = ArrayAttr(shape)
        super().__init__([shape])


@irdl_attr_definition
class TupleType(ParametrizedAttribute, TypeAttribute):
    name = "transfer.tuple"
    fields: ParameterDef[ArrayAttr[Attribute]]

    def get_num_fields(self) -> int:
        return len(self.fields.data)

    def get_fields(self):
        return [i for i in self.fields.data]

    def __init__(self, shape: list[Attribute] | ArrayAttr[Attribute]) -> None:
        if isinstance(shape, list):
            shape = ArrayAttr(shape)
        super().__init__([shape])


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


@irdl_op_definition
class SelectOp(IRDLOperation):
    """
    Select between two values based on a condition.
    """

    name = "transfer.select"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

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
    traits = traits_def(lambda: frozenset([IsTerminator(), HasParent(ConstRangeForOp)]))


@irdl_op_definition
class ConstRangeForOp(IRDLOperation):
    name = "transfer.const_range_for"

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

    lb: Operand = operand_def(T)
    ub: Operand = operand_def(T)
    step: Operand = operand_def(T)

    iter_args: VarOperand = var_operand_def(AnyAttr())

    res: OpResult = result_def(AbstractValueType)

    body: Region = region_def("single_block")

    traits = frozenset([SingleBlockImplicitTerminator(NextLoopOp)])

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

    T = Annotated[TransIntegerType | IntegerType, ConstraintVar("T")]

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
        ShlOp,
        AShrOp,
        LShrOp,
        CountLOneOp,
        CountLZeroOp,
        CountROneOp,
        CountRZeroOp,
        SetHighBitsOp,
        SetLowBitsOp,
        SetSignBitOp,
        GetLowBitsOp,
        GetBitWidthOp,
        SMinOp,
        SMaxOp,
        UMaxOp,
        UMinOp,
        UMulOverflowOp,
        ShlOverflowOp,
        SelectOp,
        IsPowerOf2Op,
        IsAllOnesOp,
        ConcatOp,
        RepeatOp,
        ExtractOp,
        ConstRangeForOp,
        NextLoopOp,
        GetAllOnesOp,
        IntersectsOp,
        AddPoisonOp,
        RemovePoisonOp,
        ReverseBitsOp,
    ],
    [TransIntegerType, AbstractValueType, TupleType],
)
