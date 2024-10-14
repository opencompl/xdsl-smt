from xdsl.ir import ParametrizedAttribute, TypeAttribute, Dialect
from xdsl.irdl import (
    irdl_attr_definition,
)


@irdl_attr_definition
class StateType(TypeAttribute, ParametrizedAttribute):
    """
    Effect state type.
    This type is used to represent the state of all effects in a program.
    """

    name = "effect.state"


EffectDialect = Dialect("effect", [], [StateType])
