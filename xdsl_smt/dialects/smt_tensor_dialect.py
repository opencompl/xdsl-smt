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
        super().__init__([shape, element_type, encoding])

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> AttributeCovT:
        return self.element_type

AnySMTTensorType: TypeAlias = SMTTensorType[Attribute]


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


class TensorExtractOp(IRDLOperation, abc.ABC):

    T: ClassVar = VarConstraint("T", irdl_to_attr_constraint(AnySMTTensorType))

    tensor = operand_def(T)
    indices = var_operand_def(BitVectorType)
    result = result_def()

    def __init__(self,  tensor: Operand, indices: Sequence[SSAValue | Operation]):
        super().__init__(operands=(tensor,indices), result_types=(tensor.type.element_type,))


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


SMTTensorDialect = Dialect(
    "smt_tensor",
    [
        TensorAndOp,
        TensorAddOp,
        TensorMultiplyOp,
        TensorAbsOp,
        TensorTransposeOp,
        TensorSubtractOp,
        TensorExtractOp,
    ],
    [SMTTensorType],
)
