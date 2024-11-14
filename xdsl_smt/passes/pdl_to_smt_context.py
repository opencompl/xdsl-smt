from xdsl.ir import Attribute, SSAValue
from typing import Sequence
from dataclasses import dataclass, field


@dataclass
class PDLToSMTRewriteContext:
    matching_effect_state: SSAValue
    rewriting_effect_state: SSAValue
    pdl_types_to_types: dict[SSAValue, Attribute] = field(default_factory=dict)
    pdl_op_to_values: dict[SSAValue, Sequence[SSAValue]] = field(default_factory=dict)
    preconditions: list[SSAValue] = field(default_factory=list)
    pdl_types_to_width: dict[SSAValue, SSAValue] = field(default_factory=dict)
