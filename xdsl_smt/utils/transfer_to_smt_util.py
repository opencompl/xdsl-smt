from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter

from xdsl_smt.dialects import smt_bitvector_dialect as smt_bv
from xdsl_smt.dialects import smt_dialect as smt


def get_constant_with_bit_vector(val: int, bv: smt_bv.BitVectorType) -> list[Operation]:
    return [smt_bv.ConstantOp(val, bv.width)]


def get_all_ones(bv: smt_bv.BitVectorType) -> list[Operation]:
    return [smt_bv.ConstantOp((1 << bv.width.data) - 1, bv.width)]


def get_zero(bv: smt_bv.BitVectorType) -> list[Operation]:
    return [smt_bv.ConstantOp(0, bv.width)]


def get_bits_constant(low_bits: SSAValue, high_bits: SSAValue) -> list[Operation]:
    """
    This is the core operation for Get/Set/Clear Low/HighBits operations.
    Given a range [low_bits, high_bits), this function returns a constant where
    0....01.............10....0
         |              |
       high_bits    low_bits
       (exclusive)  (inclusive)

    Examples: [0, 8) -> 0xff, [0, 5)] -> 0x0f
    This function implement by first getting a 0xf....f (all ones) bit vector and shifting the all_one bit vector
    """
    assert isinstance(low_bits.type, smt_bv.BitVectorType)
    width = low_bits.type.width.data
    result: list[Operation] = []
    result += get_constant_with_bit_vector(width, low_bits.type)
    width_bv = result[-1].results[0]
    result.append(smt_bv.SubOp(width_bv, high_bits))
    # Get the number of zeros expected counting from high_bits
    new_high_bits = result[-1].results[0]
    result += get_all_ones(low_bits.type)
    all_ones_bv = result[-1].results[0]
    # 1....1 -> 0...0(low_bits)1...1
    result.append(smt_bv.AShrOp(all_ones_bv, low_bits))
    # 0...0(low_bits)1...1 -> 1...10...0(low_bits)
    result.append(smt_bv.ShlOp(result[-1].results[0], low_bits))
    # 1...10...0(low_bits) -> 1...10...0(low_bits + new_high_bits)
    result.append(smt_bv.ShlOp(result[-1].results[0], new_high_bits))
    # 1...10...0(low_bits + new_high_bits) -> 0...0(new_high_bits)1...10...0(low_bits)
    result.append(smt_bv.LShrOp(result[-1].results[0], new_high_bits))
    return result


def get_low_bits_constant(low_bits: SSAValue) -> list[Operation]:
    """
    get_low_bits_constant(3) ->0b 0000 0111 -> get_bits_constants(0, 3)
    get_low_bits_constant(8) ->0b 1111 1111 -> get_bits_constants(0, 8)
    """
    assert isinstance(low_bits.type, smt_bv.BitVectorType)
    zero_op = get_zero(low_bits.type)
    return zero_op + get_bits_constant(zero_op[0].results[0], low_bits)


def get_high_bits_constant(high_bits: SSAValue) -> list[Operation]:
    """
    get_high_bits_constant(3) ->0b 1110 0000 -> get_bits_constants(8 - 3, 8)
    get_high_bits_constant(8) ->0b 1111 1111 -> get_bits_constants(8 - 8, 8)
    """
    assert isinstance(high_bits.type, smt_bv.BitVectorType)
    width = high_bits.type.width.data
    result = get_constant_with_bit_vector(width, high_bits.type)
    width_constant = result[0].results[0]
    result.append(smt_bv.SubOp(width_constant, high_bits))
    return result + get_bits_constant(result[-1].results[0], width_constant)


def get_low_bits(b: SSAValue, low_bits: SSAValue) -> list[Operation]:
    """
    get_low_bits(x, low_bits) -> x & (get_low_bits_constant(low_bits))
    """
    result = get_low_bits_constant(low_bits)
    result.append(smt_bv.AndOp(result[-1].results[0], b))
    return result


def get_high_bits(b: SSAValue, low_bits: SSAValue) -> list[Operation]:
    """
    get_high_bits(x, high_bits) -> x & (get_high_bits_constant(high_bits))
    """
    result = get_high_bits_constant(low_bits)
    result.append(smt_bv.AndOp(result[-1].results[0], b))
    return result


def count_ones(b: SSAValue) -> list[Operation]:
    assert isinstance(b.type, smt_bv.BitVectorType)
    n = b.type.width.data
    bits: list[Operation] = [smt_bv.ExtractOp(b, i, i) for i in range(n)]
    zero = smt_bv.ConstantOp(0, n - 1)
    bvs = [smt_bv.ConcatOp(zero.results[0], b.results[0]) for b in bits]
    if n == 1:
        return bits + [zero] + bvs
    result = bvs[0].res
    nb: list[Operation] = []
    for i in range(1, n):
        nb.append(smt_bv.AddOp(result, bits[i].results[0]))
        result = nb[-1].results[0]
    """
    bits = [Extract(i, i, b) for i in range(n)]
    bvs = [Concat(BitVecVal(0, n - 1), b) for b in bits]
    nb = reduce(lambda x, y: x + y, bvs)
    """
    return bits + [zero] + bvs + nb


def reverse_bits(bits: SSAValue, rewriter: PatternRewriter) -> SSAValue:
    assert isinstance(bits.type, smt_bv.BitVectorType)
    n = bits.type.width.data
    if n == 1:
        # If width is only one, no need to reverse bit
        return bits
    else:
        bits_ops: list[Operation] = [smt_bv.ExtractOp(bits, i, i) for i in range(n)]
        cur_bits: SSAValue = bits_ops[0].results[0]
        result: list[smt_bv.ConcatOp] = []
        for bit in bits_ops[1:]:
            result.append(smt_bv.ConcatOp(cur_bits, bit.results[0]))
            cur_bits = result[-1].res
        rewriter.insert_op_before_matched_op(bits_ops + result)
        return result[-1].res


pow2 = [2**i for i in range(0, 9)]


def get_leftmost_bit(b: SSAValue) -> list[Operation]:
    assert isinstance(b.type, smt_bv.BitVectorType)
    """
    bits = [b >> i for i in pow2 if i < b.size()]
    bits.append(b)
    or_bits = reduce(lambda a, b: a | b, bits)
    return or_bits - (LShR(or_bits, 1))
    """
    const_op: list[Operation] = []
    bits: list[Operation] = []
    width = b.type.width.data
    for i in pow2:
        if i < width:
            const_op.append(smt_bv.ConstantOp(i, width))
            bits.append(smt_bv.AShrOp(b, const_op[-1].results[0]))
    or_bits_res = b
    or_bits: list[Operation] = []
    for bit in bits:
        or_bits.append(smt_bv.OrOp(or_bits_res, bit.results[0]))
        or_bits_res = or_bits[-1].results[0]
    const_one = get_constant_with_bit_vector(1, b.type)
    lshr = smt_bv.LShrOp(or_bits_res, const_one[0].results[0])
    sub = smt_bv.SubOp(or_bits_res, lshr.results[0])
    return const_op + bits + or_bits + const_one + [lshr, sub]


def count_lzeros(b: SSAValue) -> list[Operation]:
    """
    name = str(b)
    tmp_count_lzeros = BitVec("tmp_" + name + "_count_lzeros", width)
    leftmostBit = get_leftmost_bit(b, solver)
    constraint:
        if b == 0 then tmp_count_lzeros == -1 (get_all_ones)
        else leftmostBit == 1 << tmp_count_lzeros
    solver.add(constraint)
    return b.size() - 1 - tmp_count_lzeros
    """
    assert isinstance(b.type, smt_bv.BitVectorType)
    tmp_count_lzeros = smt.DeclareConstOp(b.type)
    constant_one = get_constant_with_bit_vector(1, b.type)[0]
    const_zero = get_constant_with_bit_vector(0, b.type)[0]
    b_eq_0 = smt.EqOp(const_zero.results[0], b)
    neg_one_op = get_all_ones(b.type)[0]
    neg_one_eq_lzeros = smt.EqOp(neg_one_op.results[0], tmp_count_lzeros.results[0])
    true_branch_constraint = [neg_one_op, neg_one_eq_lzeros]
    leftMostBit_op = get_leftmost_bit(b)
    leftMostBit = leftMostBit_op[-1].results[0]
    false_branch_constraint: list[Operation] = [
        smt_bv.ShlOp(constant_one.results[0], tmp_count_lzeros.results[0])
    ]
    false_branch_constraint.append(
        smt.EqOp.get(leftMostBit, false_branch_constraint[-1].results[0])
    )
    ite_op = smt.IteOp(
        b_eq_0.results[0],
        true_branch_constraint[-1].results[0],
        false_branch_constraint[-1].results[0],
    )
    assert_op = smt.AssertOp(ite_op.res)
    constant_width = get_constant_with_bit_vector(b.type.width.data, b.type)[0]
    width_minus_one = smt_bv.SubOp(constant_width.results[0], constant_one.results[0])
    res = smt_bv.SubOp(width_minus_one.results[0], tmp_count_lzeros.results[0])
    return (
        [tmp_count_lzeros]
        + [const_zero, constant_one, b_eq_0]
        + true_branch_constraint
        + leftMostBit_op
        + false_branch_constraint
        + [ite_op, assert_op]
        + [constant_width, width_minus_one, res]
    )


def count_rzeros(b: SSAValue) -> tuple[list[Operation], list[Operation]]:
    """
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
    """
    assert isinstance(b.type, smt_bv.BitVectorType)
    tmp_count_rzeros = smt.DeclareConstOp(b.type)
    const_zero = get_constant_with_bit_vector(0, b.type)
    b_eq_0 = smt.EqOp(const_zero[0].results[0], b)

    const_width = get_constant_with_bit_vector(b.type.width.data, b.type)
    width_eq_rzeros = smt.EqOp(const_width[0].results[0], tmp_count_rzeros.results[0])

    const_one = get_constant_with_bit_vector(1, b.type)
    b_minus_one = smt_bv.SubOp(b, const_one[0].results[0])
    b_and_b_minus_one = smt_bv.AndOp(b, b_minus_one.results[0])
    b_minus_and = smt_bv.SubOp(b, b_and_b_minus_one.results[0])
    one_shl_tmp = smt_bv.ShlOp(const_one[0].results[0], tmp_count_rzeros.results[0])
    false_eq = smt.EqOp(b_minus_and.results[0], one_shl_tmp.results[0])

    iteOp = smt.IteOp(
        b_eq_0.results[0], width_eq_rzeros.results[0], false_eq.results[0]
    )
    assertOp = smt.AssertOp.get(iteOp.res)
    return (
        [tmp_count_rzeros],
        const_zero
        + [b_eq_0]
        + const_width
        + [width_eq_rzeros]
        + const_one
        + [
            b_minus_one,
            b_and_b_minus_one,
            b_minus_and,
            one_shl_tmp,
            false_eq,
            iteOp,
            assertOp,
        ],
    )


def count_lones(b: SSAValue) -> list[Operation]:
    neg_b = smt_bv.NotOp.get(b)
    return [neg_b] + count_lzeros(neg_b.results[0])


def count_rones(b: SSAValue) -> tuple[list[Operation], list[Operation]]:
    neg_b = smt_bv.NotOp.get(b)
    tmpRes = count_rzeros(neg_b.results[0])
    return [neg_b] + tmpRes[0], tmpRes[1]


def is_non_negative(val: SSAValue) -> list[Operation]:
    assert isinstance(val_type := val.type, smt_bv.BitVectorType)
    width = val_type.width
    const_zero = smt_bv.ConstantOp(0, width)
    val_sge_zero = smt_bv.SgeOp(val, const_zero.res)
    return [const_zero, val_sge_zero]


def is_negative(val: SSAValue) -> list[Operation]:
    assert isinstance(val_type := val.type, smt_bv.BitVectorType)
    width = val_type.width
    const_zero = smt_bv.ConstantOp(0, width)
    val_slt_zero = smt_bv.SltOp(val, const_zero.res)
    return [const_zero, val_slt_zero]
