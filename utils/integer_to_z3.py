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


def get_constant_with_bit_vector(val: int, b: BitVecRef):
    return BitVecVal(val, b.size())


def get_all_ones(width: int):
    return BitVecVal((1 << width) - 1, width)


def get_zero(width: int):
    return BitVecVal(0, width)


def get_bits_constant(low_bits: int, high_bits: int, width: int):
    high_bits = width - high_bits
    return LShR(((get_all_ones(width) >> low_bits) << low_bits) << high_bits, high_bits)


def get_low_bits_constant(low_bits: int, width: int):
    return get_bits_constant(0, low_bits, width)


def get_high_bits_constant(high_bits: int, width: int):
    return get_bits_constant(width - high_bits, width, width)


def get_low_bits(b: BitVecRef, low_bits: int):
    return get_low_bits_constant(low_bits, b.size()) & b


def set_high_bits(b: BitVecRef, high_bits: int):
    return get_high_bits_constant(high_bits, b.size()) | b


def count_ones(b: BitVecRef):
    n = b.size()
    bits = [Extract(i, i, b) for i in range(n)]
    bvs = [Concat(BitVecVal(0, n - 1), b) for b in bits]
    nb = reduce(lambda x, y: x + y, bvs)
    return nb


pow2 = [2**i for i in range(0, 9)]


def get_leftmost_bit(b: BitVec, solver: Solver):
    bits = [b >> i for i in pow2 if i < b.size()]
    bits.append(b)
    or_bits = reduce(lambda a, b: a | b, bits)
    return or_bits - (LShR(or_bits, 1))


def count_lzeros(b: BitVecRef, solver: Solver, width: int):
    name = str(b)
    tmp_count_lzeros = BitVec("tmp_" + name + "_count_lzeros", width)
    leftmostBit = get_leftmost_bit(b, solver)
    solver.add(leftmostBit == 1 << tmp_count_lzeros)
    return b.size() - 1 - tmp_count_lzeros


def count_rzeros(b: BitVecRef, solver: Solver, width: int):
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


def count_lones(b: BitVecRef, solver: Solver, width: int):
    return count_lzeros(~b, solver, width)


def count_rones(b: BitVecRef, solver: Solver, width: int):
    return count_rzeros(~b, solver, width)


def smin(a: BitVecRef, b: BitVecRef, solver: Solver):
    return If(a < b, a, b)


def smax(a: BitVecRef, b: BitVecRef, solver: Solver):
    return If(a < b, b, a)


def umin(a: BitVecRef, b: BitVecRef, solver: Solver):
    return If(ULE(a, b), a, b)


def umax(a: BitVecRef, b: BitVecRef, solver: Solver):
    return If(ULE(a, b), b, a)


def umul_overflow(a: BitVecRef, b: BitVecRef, solver: Solver):
    x, y = z3.z3._coerce_exprs(a, b)
    return Not(
        BoolRef(z3.Z3_mk_bvmul_no_overflow(x.ctx_ref(), x.as_ast(), y.as_ast(), False))
    )
