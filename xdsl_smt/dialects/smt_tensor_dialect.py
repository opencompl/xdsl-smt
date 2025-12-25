import abc
from typing import Generic, TypeAlias, ClassVar
from collections.abc import Iterable, Sequence
from xdsl.dialects.builtin import (
    IntegerAttr,
    ShapedType,
    ContainerType,
    ArrayAttr,
    IntAttr,
    NoneAttr,
)
from xdsl.irdl import (
    prop_def,
    operand_def,
    result_def,
    irdl_attr_definition,
    irdl_to_attr_constraint,
    irdl_op_definition,
    IRDLOperation,
    Attribute,
    VarConstraint,
    var_operand_def,
)
from xdsl.ir import (
    Dialect,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
    AttributeCovT,
)
from xdsl.utils.hints import isa
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType


INDEX_WIDTH = 64
IndexType = BitVectorType(INDEX_WIDTH)


@irdl_attr_definition
class SMTTensorType(
    Generic[AttributeCovT],
    ParametrizedAttribute,
    TypeAttribute,
    ShapedType,
    ContainerType[AttributeCovT],
):
    name = "smt.tensor.tensor"

    shape: ArrayAttr[IntegerAttr]
    element_type: AttributeCovT
    encoding: Attribute

    def __init__(
        self,
        element_type: AttributeCovT,
        shape: Iterable[int] | Iterable[IntegerAttr],
        encoding: Attribute = NoneAttr(),
    ):
        shape = ArrayAttr(
            [
                IntegerAttr.from_int_and_width(dim, INDEX_WIDTH)
                if isinstance(dim, int)
                else dim
                for dim in shape
            ]
        )
        super().__init__(shape, element_type, encoding)

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.value.data for i in self.shape.data)

    def get_element_type(self) -> AttributeCovT:
        return self.element_type


AnySMTTensorType: TypeAlias = SMTTensorType[Attribute]


def to_integer_array_attr(
    int_list: Iterable[int] | Iterable[IntegerAttr],
) -> ArrayAttr[IntegerAttr]:
    """
    Constructs an ArrayAttr of IntegerAttr elements from an iterable of ints
    or IntegerAttr objects. Each int is converted to an IntegerAttr with
    a width of 64 bits if necessary.
    """
    attr_list = [
        x if isinstance(x, IntegerAttr) else IntegerAttr.from_int_and_width(x, 64)
        for x in int_list
    ]
    return ArrayAttr(attr_list)


def to_int(x: int | IntegerAttr | IntAttr) -> int:
    if isinstance(x, IntegerAttr):
        return x.value.data
    elif isinstance(x, IntAttr):
        return x.data
    return x


def to_tuple_int(
    array_attr: Iterable[IntegerAttr] | Iterable[IntAttr] | Iterable[int],
) -> tuple[int, ...]:
    """
    Converts an ArrayAttr of IntegerAttr elements into a tuple of integers.
    This function extracts the integer data from each IntegerAttr in the
    provided ArrayAttr and returns it as a tuple of ints.
    """
    value_list = [to_int(x) for x in array_attr]
    return tuple(value_list)


class ElementwiseBinaryOperation(IRDLOperation, abc.ABC):
    T: ClassVar = VarConstraint("T", irdl_to_attr_constraint(AnySMTTensorType))

    lhs = operand_def(T)
    rhs = operand_def(T)

    result = result_def(T)

    def __init__(
        self, lhs: SSAValue, rhs: SSAValue, result_type: Attribute | None = None
    ):
        if result_type is None:
            result_type = lhs.type
        super().__init__(operands=(lhs, rhs), result_types=(result_type,))


class ElementwiseUnaryOperation(IRDLOperation, abc.ABC):
    T: ClassVar = VarConstraint("T", irdl_to_attr_constraint(AnySMTTensorType))

    op = operand_def(T)
    result = result_def(T)

    def __init__(self, op: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            result_type = op.type
        super().__init__(operands=(op,), result_types=(result_type,))


@irdl_op_definition
class TensorExtractOp(IRDLOperation):
    name = "smt.tensor.extract"

    tensor = operand_def(AnySMTTensorType)
    indices = var_operand_def(BitVectorType)
    result = result_def()

    def __init__(
        self,
        tensor: SSAValue,
        indices: Sequence[SSAValue | Operation],
        result_type: Attribute | None = None,
    ):
        if result_type is None:
            tensor_type = tensor.type
            assert isa(tensor_type, SMTTensorType)
            result_type = tensor_type.element_type
            assert isinstance(result_type, Attribute)
        super().__init__(operands=(tensor, indices), result_types=(result_type,))


@irdl_op_definition
class TensorAddOp(ElementwiseBinaryOperation):
    """
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
    `result` tensor. Depending on the element type, does the following:

    * For integers: integer addition.

    """

    name = "smt.tensor.add"


@irdl_op_definition
class TensorSubtractOp(ElementwiseBinaryOperation):
    """
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
    `result` tensor. Depending on the element type, does the following:

    * For integers: integer addition.

    """

    name = "smt.tensor.subtract"


@irdl_op_definition
class TensorAndOp(ElementwiseBinaryOperation):
    """
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
    `result` tensor. Depending on the element type, does the following:

    * For integers: integer addition.

    """

    name = "smt.tensor.and"


@irdl_op_definition
class TensorMultiplyOp(ElementwiseBinaryOperation):
    """
    Performs element-wise addition of two tensors `lhs` and `rhs` and produces a
    `result` tensor. Depending on the element type, does the following:

    * For integers: integer addition.

    """

    name = "smt.tensor.multiply"


@irdl_op_definition
class TensorAbsOp(ElementwiseUnaryOperation):
    """
    Performs element-wise abs operation on operand tensor and produces a result tensor.
    Depending on the element type, does the following:

    * For signed integers: integer modulus.

    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs
    """

    name = "smt.tensor.abs"


@irdl_op_definition
class TensorTransposeOp(IRDLOperation):
    """
    Performs element-wise abs operation on operand tensor and produces a result tensor.
    Depending on the element type, does the following:

    * For signed integers: integer modulus.

    https://github.com/openxla/stablehlo/blob/main/docs/spec.md#abs
    """

    name = "smt.tensor.transpose"

    operand = operand_def(AnySMTTensorType)
    result = result_def(AnySMTTensorType)
    permutation = prop_def(ArrayAttr[IntegerAttr])

    def __init__(
        self,
        operand: SSAValue,
        permutation: Iterable[int] | Iterable[IntegerAttr],
        result_type: Attribute,
    ):
        super().__init__(
            operands=(operand,),
            result_types=(result_type,),
            properties={"permutation": to_integer_array_attr(permutation)},
        )

    def get_permutation(self) -> tuple[int, ...]:
        return to_tuple_int(self.permutation)


@irdl_op_definition
class TensorPadOp(IRDLOperation):
    """
    Performs tensor pad operation
    """

    name = "smt.tensor.pad"

    operand = operand_def(AnySMTTensorType)
    padding_value = operand_def(Attribute)
    result = result_def(AnySMTTensorType)
    edge_padding_low = prop_def(ArrayAttr[IntegerAttr])
    edge_padding_high = prop_def(ArrayAttr[IntegerAttr])
    interior_padding = prop_def(ArrayAttr[IntegerAttr])

    def get_result_shape(
        self,
        operand_type: Attribute,
        edge_padding_low: Iterable[int] | Iterable[IntegerAttr],
        edge_padding_high: Iterable[int] | Iterable[IntegerAttr],
        interior_padding: Iterable[int] | Iterable[IntegerAttr],
    ) -> SMTTensorType[Attribute]:
        assert isa(operand_type, SMTTensorType)
        shape = operand_type.get_shape()
        padding_low = to_tuple_int(edge_padding_low)
        padding_high = to_tuple_int(edge_padding_high)
        padding_inner = to_tuple_int(interior_padding)
        assert len(shape) == len(padding_low) == len(padding_high) == len(padding_inner)
        new_shape: list[int] = []
        for i in range(0, len(shape)):
            new_shape.append(
                padding_low[i]
                + (shape[i] - 1) * padding_inner[i]
                + shape[i]
                + padding_high[i]
            )
        return SMTTensorType(operand_type.element_type, new_shape)

    def __init__(
        self,
        operand: SSAValue,
        padding_value: SSAValue,
        edge_padding_low: Iterable[int] | Iterable[IntegerAttr],
        edge_padding_high: Iterable[int] | Iterable[IntegerAttr],
        interior_padding: Iterable[int] | Iterable[IntegerAttr],
    ):
        result_type = self.get_result_shape(
            operand.type, edge_padding_low, edge_padding_high, interior_padding
        )
        super().__init__(
            operands=(operand, padding_value),
            result_types=(result_type,),
            properties={
                "edge_padding_low": to_integer_array_attr(edge_padding_low),
                "edge_padding_high": to_integer_array_attr(edge_padding_high),
                "interior_padding": to_integer_array_attr(interior_padding),
            },
        )

    def get_edge_padding_low(self) -> tuple[int, ...]:
        return to_tuple_int(self.edge_padding_low)

    def get_edge_padding_high(self) -> tuple[int, ...]:
        return to_tuple_int(self.edge_padding_high)

    def get_interior_padding(self) -> tuple[int, ...]:
        return to_tuple_int(self.interior_padding)


@irdl_op_definition
class TensorSliceOp(IRDLOperation):
    """
    Performs tensor slice operation
    """

    name = "smt.tensor.slice"

    operand = operand_def(AnySMTTensorType)
    result = result_def(AnySMTTensorType)
    start_indices = prop_def(ArrayAttr[IntegerAttr])
    limit_indices = prop_def(ArrayAttr[IntegerAttr])
    strides = prop_def(ArrayAttr[IntegerAttr])

    def get_result_shape(
        self,
        operand_type: Attribute,
        start_indices: Iterable[int] | Iterable[IntegerAttr],
        limit_indices: Iterable[int] | Iterable[IntegerAttr],
        strides: Iterable[int] | Iterable[IntegerAttr],
    ) -> SMTTensorType[Attribute]:
        assert isa(operand_type, SMTTensorType)
        start_indices = to_tuple_int(start_indices)
        limit_indices = to_tuple_int(limit_indices)
        strides = to_tuple_int(strides)
        assert len(start_indices) == len(limit_indices) == len(strides)
        new_shape: list[int] = []
        for i in range(0, len(strides)):
            new_shape.append((limit_indices[i] - start_indices[i]) // strides[i])
            assert new_shape[-1] > 0
        return SMTTensorType(operand_type.element_type, new_shape)

    def __init__(
        self,
        operand: SSAValue,
        start_indices: Iterable[int] | Iterable[IntegerAttr],
        limit_indices: Iterable[int] | Iterable[IntegerAttr],
        strides: Iterable[int] | Iterable[IntegerAttr],
    ):
        result_type = self.get_result_shape(
            operand.type, start_indices, limit_indices, strides
        )
        super().__init__(
            operands=(operand,),
            result_types=(result_type,),
            properties={
                "start_indices": to_integer_array_attr(start_indices),
                "limit_indicts": to_integer_array_attr(limit_indices),
                "strides": to_integer_array_attr(strides),
            },
        )

    def get_start_indices(self) -> tuple[int, ...]:
        return to_tuple_int(self.start_indices)

    def get_limit_indicts(self) -> tuple[int, ...]:
        return to_tuple_int(self.limit_indices)

    def get_strides(self) -> tuple[int, ...]:
        return to_tuple_int(self.strides)


SMTTensorDialect = Dialect(
    "smt.tensor",
    [
        TensorAndOp,
        TensorAddOp,
        TensorMultiplyOp,
        TensorAbsOp,
        TensorTransposeOp,
        TensorSubtractOp,
        TensorExtractOp,
        TensorPadOp,
        TensorSliceOp,
    ],
    [SMTTensorType],
)
