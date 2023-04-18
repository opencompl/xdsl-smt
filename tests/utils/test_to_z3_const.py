from xdsl.dialects.builtin import UnregisteredOp
from dialects.smt_bitvector_dialect import BitVectorType
from dialects.smt_dialect import BoolType

from utils.z3_to_dialect import to_z3_consts


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
