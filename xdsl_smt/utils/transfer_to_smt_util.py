from xdsl.ir import Operation, SSAValue

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


def clear_low_bits(b: SSAValue, low_bits: SSAValue) -> list[Operation]:
    """
    clear_low_bits(x, low_bits) -> x & ~(get_low_bits_constant(low_bits))
    """
    result = get_low_bits_constant(low_bits)
    result.append(smt_bv.NotOp(result[-1].results[0]))
    result.append(smt_bv.AndOp(result[-1].results[0], b))
    return result


def clear_high_bits(b: SSAValue, low_bits: SSAValue) -> list[Operation]:
    """
    clear_high_bits(x, high_bits) -> x & ~(get_high_bits_constant(high_bits))
    """
    result = get_high_bits_constant(low_bits)
    result.append(smt_bv.NotOp(result[-1].results[0]))
    result.append(smt_bv.AndOp(result[-1].results[0], b))
    return result


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


def set_high_bits(b: SSAValue, high_bits: SSAValue) -> list[Operation]:
    """
    set_high_bits(x, high_bits) -> x | (get_high_bits_constant(high_bits))
    """
    result = get_high_bits_constant(high_bits)
    result.append(smt_bv.OrOp(result[-1].results[0], b))
    return result


def set_low_bits(b: SSAValue, low_bits: SSAValue) -> list[Operation]:
    """
    set_low_bits(x, low_bits) -> x | (get_low_bits_constant(low_bits))
    """
    result = get_low_bits_constant(low_bits)
    result.append(smt_bv.OrOp(result[-1].results[0], b))
    return result


def count_zero_side_bits(b: SSAValue, from_left: bool) -> list[Operation]:
    """Count zero bits from the left or right as a pure bitvector term."""

    assert isinstance(bv_type := b.type, smt_bv.BitVectorType)
    width = bv_type.width.data
    ops: list[Operation] = []
    const_cache: dict[int, smt_bv.ConstantOp] = {}

    def get_const(value: int) -> smt_bv.ConstantOp:
        value %= 1 << width
        if value not in const_cache:
            const_cache[value] = smt_bv.ConstantOp(value, bv_type.width)
            ops.append(const_cache[value])
        return const_cache[value]

    def descending_powers_of_two() -> list[int]:
        step = 1
        while step < width:
            step <<= 1
        step >>= 1

        result: list[int] = []
        while step != 0:
            result.append(step)
            step >>= 1
        return result

    zero = get_const(0)
    width_bv = get_const(width)

    if from_left:
        count = zero.results[0]
        current = b

        for step in descending_powers_of_two():
            top_mask = ((1 << step) - 1) << (width - step)
            top_mask_bv = get_const(top_mask)
            step_bv = get_const(step)

            masked = smt_bv.AndOp(current, top_mask_bv.results[0])
            is_zero = smt.EqOp(masked.results[0], zero.results[0])
            next_count = smt_bv.AddOp(count, step_bv.results[0])
            shifted = smt_bv.ShlOp(current, step_bv.results[0])
            count_ite = smt.IteOp(is_zero.results[0], next_count.results[0], count)
            current_ite = smt.IteOp(is_zero.results[0], shifted.results[0], current)

            ops.extend([masked, is_zero, next_count, shifted, count_ite, current_ite])
            count = count_ite.results[0]
            current = current_ite.results[0]

        input_is_zero = smt.EqOp(b, zero.results[0])
        result = smt.IteOp(input_is_zero.results[0], width_bv.results[0], count)
        ops.extend([input_is_zero, result])
        return ops

    minus_b = smt_bv.SubOp(zero.results[0], b)
    isolated = smt_bv.AndOp(b, minus_b.results[0])
    input_is_zero = smt.EqOp(b, zero.results[0])
    count = smt.IteOp(
        input_is_zero.results[0], width_bv.results[0], get_const(width - 1).results[0]
    )
    ops.extend([minus_b, isolated, input_is_zero, count])

    for step in descending_powers_of_two():
        mask = 0
        block = step << 1
        for start in range(0, width, block):
            low_chunk = min(step, width - start)
            mask |= ((1 << low_chunk) - 1) << start

        mask_bv = get_const(mask)
        step_bv = get_const(step)

        masked = smt_bv.AndOp(isolated.results[0], mask_bv.results[0])
        masked_is_zero = smt.EqOp(masked.results[0], zero.results[0])
        next_count = smt_bv.SubOp(count.results[0], step_bv.results[0])
        count = smt.IteOp(
            masked_is_zero.results[0], count.results[0], next_count.results[0]
        )
        ops.extend([masked, masked_is_zero, next_count, count])

    return ops
