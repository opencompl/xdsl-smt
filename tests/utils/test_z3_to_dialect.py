from typing import Any
import pytest

from z3 import And, BitVecSort, BoolSort, Bools, Exists, ForAll, Or, Xor

from xdsl.dialects.builtin import UnregisteredOp

from dialects.smt_bitvector_dialect import BitVectorType
from dialects.smt_dialect import (AndOp, BinaryBoolOp, BinaryTOp, BoolType,
                                  EqOp, ExistsOp, ForallOp, OrOp, XorOp,
                                  YieldOp, DistinctOp)
from utils.z3_to_dialect import to_z3_consts, z3_sort_to_dialect, z3_to_dialect


def test_to_z3_consts():
    op = UnregisteredOp.with_name('test.test').create(result_types=[
        BoolType(),
        BoolType(),
        BitVectorType(15),
        BitVectorType(7)
    ])
    val1, val2, val3, val4 = op.results

    (v1, v2, v2p, v3, v4, v4p) = to_z3_consts(val1, val2, val2, val3, val4,
                                              val4)
    assert v1 is not v2
    assert v2 is v2p
    assert v3 is not v4
    assert v4 is v4p


def test_z3_sort_to_dialect():
    assert z3_sort_to_dialect(BoolSort()) == BoolType()
    assert z3_sort_to_dialect(BitVecSort(10)) == BitVectorType(10)
    assert z3_sort_to_dialect(BitVecSort(42)) == BitVectorType(42)


def test_z3_to_dialect_const():
    op = UnregisteredOp.with_name('test.test').create(result_types=[
        BoolType(),
    ])
    val = op.results[0]

    v, = to_z3_consts(val)
    ops, val_prime = z3_to_dialect(v)

    assert ops == []
    assert val_prime is val


def test_z3_to_dialect_quant():
    x, y = Bools('x y')
    forall = ForAll([x, y], x)  # type: ignore
    exists = Exists([x, y], x)  # type: ignore

    ops, var = z3_to_dialect(forall)
    assert len(ops) == 1
    ops = ops[0]
    assert isinstance(ops, ForallOp)
    assert ops.res == var
    assert ops.body.block.args[0].typ == BoolType()
    assert ops.body.block.args[1].typ == BoolType()
    assert len(ops.body.block.ops) == 1
    assert isinstance(ops.body.block.ops[0], YieldOp)
    assert ops.body.block.ops[0].ret == ops.body.block.args[0]

    ops, var = z3_to_dialect(exists)
    assert len(ops) == 1
    ops = ops[0]
    assert isinstance(ops, ExistsOp)
    assert ops.res == var
    assert ops.body.block.args[0].typ == BoolType()
    assert ops.body.block.args[1].typ == BoolType()
    assert len(ops.body.block.ops) == 1
    assert isinstance(ops.body.block.ops[0], YieldOp)
    assert ops.body.block.ops[0].ret == ops.body.block.args[0]


@pytest.mark.parametrize('xdsl_op, z3_op', [(OrOp, Or), (AndOp, And),
                                            (XorOp, Xor),
                                            (EqOp, lambda x, y: x == y),
                                            (DistinctOp, lambda x, y: x != y)])
def test_z3_to_dialect_binary_core(xdsl_op: type[BinaryBoolOp | BinaryTOp],
                                   z3_op: Any):
    op = UnregisteredOp.with_name('test.test').create(result_types=[
        BoolType(),
        BoolType(),
    ])
    lhs, rhs = op.results
    x, y = to_z3_consts(lhs, rhs)

    ops, var = z3_to_dialect(z3_op(x, y))  # type: ignore
    assert len(ops) == 1
    assert isinstance(ops[0], xdsl_op)
    assert ops[0].res == var
    assert ops[0].lhs == lhs
    assert ops[0].rhs == rhs
