from xdsl.irdl import IRDLOperation
from xdsl.ir import Operation, SSAValue

from ..dialects import smt_bitvector_dialect as smt_bv
from ..dialects import smt_dialect as smt
from ..dialects import transfer
from xdsl.ir import Attribute, MLContext
from xdsl_smt.dialects.smt_dialect import BoolType

import z3
from z3 import (
    BitVecVal,
    LShR,
    Extract,
    Concat,
    BitVec,
    Solver,
    BitVecRef,
    If,
    ULE,
    Not,
    BoolRef,
)
from functools import reduce


def get_constant_with_bit_vector(val: int, bv: smt_bv.BitVectorType) -> list[Operation]:
    return [smt_bv.ConstantOp(val, bv.width)]


def get_all_ones(bv: smt_bv.BitVectorType) -> list[Operation]:
    return [smt_bv.ConstantOp((1 << bv.width.data) - 1, bv.width)]


def get_zero(bv: smt_bv.BitVectorType) -> list[Operation]:
    return [smt_bv.ConstantOp(0, bv.width)]


'''
return a bit vector with bits set in a range[low_bits, high_bits)
LShR(((get_all_ones(width) >> low_bits) << low_bits) << high_bits, high_bits)
'''


def get_bits_constant(low_bits: SSAValue, high_bits: SSAValue) -> list[Operation]:
    assert isinstance(low_bits.type, smt_bv.BitVectorType)
    width = low_bits.type.width.data
    result = []
    result += get_constant_with_bit_vector(width, low_bits.type)
    width_bv = result[-1].results[0]
    result.append(smt_bv.SubOp(width_bv, high_bits))
    new_high_bits = result[-1].results[0]
    result += get_all_ones(low_bits.type)
    all_ones_bv = result[-1].results[0]
    result.append(smt_bv.AShrOp(all_ones_bv, low_bits))
    result.append(smt_bv.ShlOp(result[-1].results[0], low_bits))
    result.append(smt_bv.ShlOp(result[-1].results[0], new_high_bits))
    result.append(smt_bv.LShrOp(result[-1].results[0], new_high_bits))
    return result


def get_low_bits_constant(low_bits: SSAValue) -> list[Operation]:
    zero_op = get_zero(low_bits.type)
    return zero_op + get_bits_constant(zero_op[0].results[0], low_bits)


def get_high_bits_constant(high_bits: SSAValue) -> list[Operation]:
    assert isinstance(high_bits.type, smt_bv.BitVectorType)
    width = high_bits.type.width.data
    result = get_constant_with_bit_vector(width, high_bits.type)
    width_constant = result[0].results[0]
    result += smt_bv.SubOp(width_constant, high_bits)
    return result + get_bits_constant(result[-1].results[0], width_constant)


def get_low_bits(b: SSAValue, low_bits: SSAValue) -> list[Operation]:
    result = get_low_bits_constant(low_bits)
    result.append(smt_bv.AndOp(result[-1].results[0], b))
    return result
    # get_low_bits_constant(low_bits, b.size()) & b


def set_high_bits(b: SSAValue, high_bits: SSAValue) -> list[Operation]:
    result = get_high_bits_constant(high_bits)
    result.append(smt_bv.OrOp(result[-1].results[0], b))
    return result
    # return get_high_bits_constant(high_bits, b.size()) | b


def count_ones(b: SSAValue) -> list[Operation]:
    assert isinstance(b.type, smt_bv.BitVectorType)
    n = b.type.width.data
    bits = [smt_bv.ExtractOp(b, i, i) for i in range(n)]
    zero = smt_bv.ConstantOp(0, n - 1)
    bvs = [smt_bv.ConcatOp(zero.results[0], b.results[0]) for b in bits]
    if n == 1:
        return bits + [zero] + bvs
    result = bvs[0].results[0]
    nb = []
    for i in range(1, n):
        nb.append(smt_bv.AddOp(result, bits[i].results[0]))
        result = nb[-1].results[0]
    '''
    bits = [Extract(i, i, b) for i in range(n)]
    bvs = [Concat(BitVecVal(0, n - 1), b) for b in bits]
    nb = reduce(lambda x, y: x + y, bvs)
    '''
    return bits + [zero] + bvs + nb


pow2 = [2 ** i for i in range(0, 9)]


def get_leftmost_bit(b: SSAValue) -> list[Operation]:
    assert isinstance(b.type, smt_bv.BitVectorType)
    '''
    bits = [b >> i for i in pow2 if i < b.size()]
    bits.append(b)
    or_bits = reduce(lambda a, b: a | b, bits)
    return or_bits - (LShR(or_bits, 1))
    '''
    const_op = []
    bits = []
    width = b.type.width.data
    for i in pow2:
        if i < width:
            const_op.append(smt_bv.ConstantOp(i, width))
            bits.append(smt_bv.AShrOp(b, const_op[-1].results[0]))
    or_bits_res = b
    or_bits = []
    for bit in bits:
        or_bits.append(smt_bv.OrOp(or_bits_res, bit.results[0]))
        or_bits_res = or_bits[-1].results[0]
    const_one = get_constant_with_bit_vector(1, b.type)
    lshr = smt_bv.LShrOp(or_bits_res, const_one[0].results[0])
    sub = smt_bv.SubOp(or_bits_res, lshr.results[0])
    return const_op + bits + or_bits + [const_one, lshr, sub]


def count_lzeros(b: SSAValue) -> list[Operation]:
    '''
    name = str(b)
    tmp_count_lzeros = BitVec("tmp_" + name + "_count_lzeros", width)
    leftmostBit = get_leftmost_bit(b, solver)
    solver.add(leftmostBit == 1 << tmp_count_lzeros)
    return b.size() - 1 - tmp_count_lzeros
    '''
    tmp_count_lzeros = smt.DeclareConstOp(b.type)
    leftMostBit_op = get_leftmost_bit(b)
    leftMostBit = leftMostBit_op[-1].results[0]
    constant_one = get_constant_with_bit_vector(1, b.type)[0]
    constraint = [smt_bv.ShlOp(constant_one.results[0], tmp_count_lzeros.results[0])]
    constraint.append(smt.EqOp(leftMostBit, constraint[-1].results[0]))
    constraint.append(smt.AssertOp(constraint[-1].results[0]))
    constant_width = get_constant_with_bit_vector(b.type.width, b.type)[0]
    width_minus_one = smt_bv.SubOp(constant_width.results[0], constant_one.results[0])
    res = smt_bv.SubOp(width_minus_one.results[0], tmp_count_lzeros.results[0])
    return [tmp_count_lzeros] + leftMostBit_op + [constant_one] + constraint + [constant_width, width_minus_one, res]


def count_rzeros(b: SSAValue) -> list[Operation]:
    '''
    name = str(b)
    tmp_count_rzeros = BitVec("tmp_" + name + "_count_rzeros", width)
    solver.add(
        If(
            b == 0,
            tmp_count_rzeros == width,
            b - (b & (b - 1)) == 1 << tmp_count_rzeros,
        )
    )
    return tmp_count_rzeros
    '''
    tmp_count_rzeros = smt.DeclareConstOp(b.type)
    const_zero = get_constant_with_bit_vector(0, b.type)
    b_eq_0 = smt.EqOp(const_zero[0].results[0], b)

    const_width = get_constant_with_bit_vector(b.type.width, b.type)
    width_eq_rzeros = smt.EqOp(const_width[0].results[0], tmp_count_rzeros.results[0])

    const_one = get_constant_with_bit_vector(1, b.type)
    b_minus_one = smt_bv.SubOp(b, const_one[0].results[0])
    b_and_b_minus_one = smt_bv.AndOp(b, b_minus_one.results[0])
    b_minus_and = smt_bv.SubOp(b, b_and_b_minus_one.results[0])
    one_shl_tmp = smt_bv.ShlOp(const_one[0].results[0], tmp_count_rzeros.results[0])
    false_eq = smt.EqOp(b_minus_and.results[0], one_shl_tmp.results[0])

    result = smt.IteOp(b_eq_0.results[0], width_eq_rzeros.results[0], false_eq.results[0])
    return ([tmp_count_rzeros] +
            const_zero + [b_eq_0] + const_width +
            [width_eq_rzeros] + const_one + [b_minus_one,
                                             b_and_b_minus_one,
                                             b_minus_and,
                                             one_shl_tmp,
                                             false_eq, result])


def count_lones(b: SSAValue) -> list[Operation]:
    neg_b = smt_bv.NegOp(b)
    return [neg_b] + count_lzeros(neg_b.results[0])


def count_rones(b: SSAValue) -> list[Operation]:
    neg_b = smt_bv.NegOp(b)
    return [neg_b] + count_rzeros(neg_b.results[0])
