from __future__ import annotations
from xdsl.ir import Attribute, Dialect, OpResult
from xdsl.irdl import OpAttr, Operand, irdl_op_definition, IRDLOperation

from traits.effects import Pure
from abc import ABC
from z3 import *

from dataclasses import dataclass, field
from enum import Enum
from xdsl.dialects.builtin import AnyIntegerAttr
from xdsl.dialects.builtin import ArrayAttr
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.builtin import IndexType
from xdsl.dialects.builtin import IntegerType
from xdsl.dialects.builtin import ContainerOf
from xdsl.dialects.builtin import TypeAttribute
from xdsl.dialects.builtin import i1
from xdsl.dialects.arith import Constant
import functools
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, List, Sequence, TypeVar, Union, Set, Optional

@irdl_op_definition
class Add(IRDLOperation, Pure):
    name = "index.add"
    lhs: Annotated[Operand, IndexType]
    rhs: Annotated[Operand, IndexType]
    result: Annotated[OpResult, IndexType]

@irdl_op_definition
class And(IRDLOperation, Pure):
    name = "index.and"
    lhs: Annotated[Operand, IndexType]
    rhs: Annotated[Operand, IndexType]
    result: Annotated[OpResult, IndexType]

@irdl_op_definition
class Cmp(IRDLOperation, Pure):
    name = "index.cmp"
    lhs: Annotated[Operand, IndexType]
    rhs: Annotated[Operand, IndexType]
    predicate: OpAttr[Attribute]
    result: Annotated[OpResult, IndexType]


@irdl_op_definition
class Constant(IRDLOperation, Pure):
    name = "index.constant"
    value: OpAttr[Attribute]
    result: Annotated[OpResult, IndexType]

Index = Dialect(
    [
        Add,
        And,
        Cmp,
        Constant,
    ],
    [],
)
