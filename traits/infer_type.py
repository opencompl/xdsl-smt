from abc import abstractmethod
from typing import Mapping, Sequence

from xdsl.ir import Attribute


class InferResultTypeInterface:
    @staticmethod
    @abstractmethod
    def infer_result_type(
        operand_types: Sequence[Attribute], attributes: Mapping[str, Attribute] = {}
    ) -> Sequence[Attribute]:
        ...
