import abc
from typing import TypeVar, IO, Generic, TypeAlias, ClassVar, Annotated, cast
from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from ..traits.effects import Pure
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    ShapedType,
    ContainerType,
    ArrayAttr,
    IntAttr,
    NoneAttr,
    DenseArrayBase,
)
from xdsl.irdl import (
    attr_def,
    prop_def,
    operand_def,
    result_def,
    irdl_attr_definition,
    irdl_to_attr_constraint,
    irdl_op_definition,
    Operand,
    IRDLOperation,
    Attribute,
    VarConstraint,
    base,
    ConstraintVar, var_operand_def,
)
from xdsl.ir import (
    Dialect,
    OpResult,
    Operation,
    ParametrizedAttribute,
    SSAValue,
    TypeAttribute,
    AttributeCovT,
)
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorType

@irdl_attr_definition
class SMTTensorType(
    Generic[AttributeCovT],
    ParametrizedAttribute,
    TypeAttribute,
    ShapedType,
    ContainerType[AttributeCovT],
):
    name = "smt.tensor"

    shape: ArrayAttr[IntAttr]
    element_type: AttributeCovT
    encoding: Attribute

    def __init__(
        self,
        element_type: AttributeCovT,
        shape: Iterable[int | IntAttr],
        encoding: Attribute = NoneAttr(),
    ):
        shape = ArrayAttr(
            [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
        )
        super().__init__(shape, element_type, encoding)

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> AttributeCovT:
        return self.element_type

AnySMTTensorType: TypeAlias = SMTTensorType[Attribute]
INDEX_WIDTH=32
IndexType = BitVectorType(INDEX_WIDTH)

def toIntegerArrayAttr(int_list: Iterable[int | IntegerAttr]) -> ArrayAttr[IntegerAttr]:
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


def toInt(x:int|IntegerAttr | IntAttr) -> int:
    if isinstance(x, IntegerAttr):
        return x.value.data
    elif isinstance(x, IntAttr):
        return x.data
    return x


def toTupleInt(array_attr: Iterable[int | IntAttr | IntegerAttr]) -> tuple[int, ...]:
    """
    Converts an ArrayAttr of IntegerAttr elements into a tuple of integers.
    This function extracts the integer data from each IntegerAttr in the
    provided ArrayAttr and returns it as a tuple of ints.
    """
    value_list = [toInt(x) for x in array_attr]
    return tuple(value_list)

class ElementwiseBinaryOperation(IRDLOperation, abc.ABC):
    # TODO: Remove this constraint for complex types.
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
    # TODO: Remove this constraint for complex types.
    T: ClassVar = VarConstraint("T", irdl_to_attr_constraint(AnySMTTensorType))

    op = operand_def(T)
    result = result_def(T)

    def __init__(self, op: SSAValue, result_type: Attribute | None = None):
        if result_type is None:
            result_type = op.type
        super().__init__(operands=(op,), result_types=(result_type,))

@irdl_op_definition
class TensorExtractOp(IRDLOperation):
    #T: ClassVar = VarConstraint("T", irdl_to_attr_constraint(AnySMTTensorType))

    name = "smt.tensor.extract"

    tensor = operand_def(AnySMTTensorType)
    indices = var_operand_def(BitVectorType)
    result = result_def()

    def __init__(self,  tensor: Operand, indices: Sequence[SSAValue | Operation], result_type: Attribute | None = None):
        if result_type is None:
            result_type = tensor.type.element_type
        super().__init__(operands=(tensor,indices), result_types=(result_type,))


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

    ElementType = Annotated[Attribute, ConstraintVar("ElementType")]

    operand = operand_def(SMTTensorType[ElementType])
    result = result_def(SMTTensorType[ElementType])
    permutation = prop_def(DenseArrayBase)

    def __init__(
        self, operand: SSAValue, permutation: DenseArrayBase, result_type: Attribute
    ):
        super().__init__(
            operands=(operand,),
            result_types=(result_type,),
            properties={"permutation": permutation},
        )

    def get_permutation(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], self.permutation.get_values())


@irdl_op_definition
class TensorPadOp(IRDLOperation):
    """
    Performs tensor pad operation
    """

    name = "smt.tensor.pad"

    ElementType = Annotated[Attribute, ConstraintVar("ElementType")]

    operand = operand_def(SMTTensorType[ElementType])
    padding_value = operand_def(ElementType)
    result = result_def(SMTTensorType[ElementType])
    edge_padding_low = prop_def(ArrayAttr[IntegerAttr])
    edge_padding_high = prop_def(ArrayAttr[IntegerAttr])
    interior_padding = prop_def(ArrayAttr[IntegerAttr])


    def get_result_shape(self, operand_type, edge_padding_low, edge_padding_high, interior_padding) -> SMTTensorType:
        assert isinstance(operand_type, SMTTensorType)
        shape = operand_type.get_shape()
        padding_low = toTupleInt(edge_padding_low)
        padding_high = toTupleInt(edge_padding_high)
        padding_inner = toTupleInt(interior_padding)
        assert len(shape) == len(padding_low) == len(padding_high) == len(padding_inner)
        new_shape = []
        for i in range(0, len(shape)):
            new_shape.append(padding_low[i]+(shape[i]-1)*interior_padding[i]+shape[i]+padding_high[i])
        return SMTTensorType(operand_type.element_type, new_shape)

    def __init__(
        self, operand: SSAValue, padding_value:SSAValue, edge_padding_low:Iterable[int |IntegerAttr],
            edge_padding_high: Iterable[int | IntegerAttr],  interior_padding:Iterable[int |IntegerAttr],
    ):
        result_type = self.get_result_shape(operand.type, edge_padding_low, edge_padding_high, interior_padding)
        super().__init__(
            operands=(operand,padding_value),
            result_types=(result_type,),
            properties={"edge_padding_low": toIntegerArrayAttr(edge_padding_low),
                        "edge_padding_high": toIntegerArrayAttr(edge_padding_high),
                        "interior_padding": toIntegerArrayAttr(interior_padding)},
        )

    def get_edge_padding_low(self) -> tuple[int, ...]:
        return toTupleInt(self.edge_padding_low)

    def get_edge_padding_high(self) -> tuple[int, ...]:
        return toTupleInt(self.edge_padding_high)

    def get_interior_padding(self) -> tuple[int, ...]:
        return toTupleInt(self.interior_padding)


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
        TensorPadOp
    ],
    [SMTTensorType],
)
