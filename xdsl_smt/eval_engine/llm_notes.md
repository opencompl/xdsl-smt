# LLM Abstract Transfer Function Synthesis Testing

## Models tested

* deepseek
* chatgpt
* gemini

## Prompt

> Write an abstract transfer function for unsigned min in the known bits domain.
> Here is the C++ function prototype that your function should follow: `std::tuple<int, int> umin(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs);`.
> The first element of the tuple represents the known zeros, and the second element represents the known ones.
> Don't include any explanation, just C++ code for the function.

## Transfer Function Scores

### `umin`

| Model    | cost   | unsound | imprecision | exact | cases |
|----------|--------|---------|-------------|-------|-------|
| DeepSeek | 0.6219 | 3768    | 1996        | 2403  | 6561  |
| Gemini   | 0.7439 | 1026    | 53370       | 717   | 6561  |
| ChatGPT  | 0.6219 | 3768    | 1996        | 2403  | 6561  |

Notes:
* DeepSeek: used DeepThink R1, which took about 6 minutes to think
* Gemini: code doensn't even compile by default, had to fix tuple access.

### `add`

| Model    | cost   | unsound | imprecision | exact | cases |
|----------|--------|---------|-------------|-------|-------|
| DeepSeek | 0.6777 | 3272    | 22629       | 1821  | 6561  |
| Gemini   | 0.8740 | 5454    | 11613       | 757   | 6561  |
| ChatGPT  | 0.5582 | 1308    | 30092       | 2310  | 6561  |

Notes:
* Gemini: still can't get the tuple access right

### `xor`

| Model    | cost   | unsound | imprecision | exact | cases |
|----------|--------|---------|-------------|-------|-------|
| DeepSeek | 0.0000 | 0       | 0           | 6561  | 6561  |
| Gemini   | 0.9282 | 5249    | 25527       | 261   | 6561  |
| ChatGPT  | 0.0000 | 0       | 0           | 6561  | 6561  |

Notes:
* Gemini: still can't get the tuple access right
* Gemini: idek what it's thought process was here, this should be easy

### `udiv`

| Model              | cost   | unsound | imprecision | exact | cases |
|--------------------|--------|---------|-------------|-------|-------|
| DeepSeek           | 0.7415 | 73      | 77299       | 498   | 6561  |
| DeepSeek Reasoning | 0.7790 | 640     | 73109       | 332   | 6561  |
| Gemini             | 0.8009 | 1860    | 53802       | 458   | 6561  |
| Gemini Reasoning   | 0.7359 | 0       | 77730       | 526   | 6561  |
| ChatGPT            | 0.7617 | 1625    | 38280       | 720   | 6561  |
| ChatGPT Reasoning  | 0.7790 | 640     | 73109       | 332   | 6561  |

Notes:
* Much harder problem so I decided to try the reasoning modes of the models
* ChatGPT and DeepSeek's reasoning modes spit out much more complicated code with many edge cases
* While Gemini's reasoning mode decides the problem is too hard and returns top (this does however produce the best score)

## Transfer Function Code

### `umin`

```cpp
// DeepSeek
std::tuple<int, int> umin(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    auto [lz, lo] = lhs;
    auto [rz, ro] = rhs;
    int known_zeros = lz | rz;
    int known_ones = lo & ro;
    return {known_zeros, known_ones};
}

// Gemini
std::tuple<int, int> umin(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    int lhs_zeros = std::get(lhs);
    int lhs_ones = std::get(lhs);
    int rhs_zeros = std::get(rhs);
    int rhs_ones = std::get(rhs);

    int result_zeros = 0;
    int result_ones = 0;

    if (lhs_ones > 0 && rhs_zeros == 0) {
        result_ones = lhs_ones;
    } else if (rhs_ones > 0 && lhs_zeros == 0) {
        result_ones = rhs_ones;
    } else if (lhs_ones == 0 && rhs_ones == 0) {
        result_zeros = std::max(lhs_zeros, rhs_zeros);
    } else {
        result_zeros = 0;
        result_ones = 0;
    }

    return std::make_tuple(result_zeros, result_ones);
}

// ChatGPT
std::tuple<int, int> umin(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    int lhsKnownZero = std::get<0>(lhs);
    int lhsKnownOne = std::get<1>(lhs);
    int rhsKnownZero = std::get<0>(rhs);
    int rhsKnownOne = std::get<1>(rhs);

    int minKnownZero = lhsKnownZero | rhsKnownZero;
    int minKnownOne = lhsKnownOne & rhsKnownOne;

    return std::make_tuple(minKnownZero, minKnownOne);
}
```

### `add`

```cpp
// DeepSeek
std::tuple<int, int> abstract_add(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    int lhs_zeros = std::get<0>(lhs);
    int lhs_ones = std::get<1>(lhs);
    int rhs_zeros = std::get<0>(rhs);
    int rhs_ones = std::get<1>(rhs);

    int known_zeros = lhs_zeros & rhs_zeros;
    int known_ones = lhs_ones & rhs_ones;

    return std::make_tuple(known_zeros, known_ones);
}

// Gemini
std::tuple<int, int> abstract_add(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    int lhs_zeros = std::get(lhs);
    int lhs_ones = std::get(lhs);
    int rhs_zeros = std::get(rhs);
    int rhs_ones = std::get(rhs);

    int result_zeros = std::min(lhs_zeros, rhs_zeros);
    int result_ones = std::min(lhs_ones, rhs_ones);

    return std::make_tuple(result_zeros, result_ones);
}

// ChatGPT
std::tuple<int, int> abstract_add(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    int lhs_zero = std::get<0>(lhs);
    int lhs_one = std::get<1>(lhs);
    int rhs_zero = std::get<0>(rhs);
    int rhs_one = std::get<1>(rhs);

    int known_zeros = (lhs_zero & rhs_zero);
    int known_ones = (lhs_one & rhs_one);

    int carry = (lhs_one & rhs_one) | ((lhs_one | rhs_one) & (lhs_zero | rhs_zero));

    known_zeros &= ~carry;
    known_ones &= ~carry;

    return {known_zeros, known_ones};
}
```

### `xor`

```cpp
// DeepSeek
std::tuple<int, int> abstract_xor(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    int lhs_zero = std::get<0>(lhs);
    int lhs_one = std::get<1>(lhs);
    int rhs_zero = std::get<0>(rhs);
    int rhs_one = std::get<1>(rhs);

    int known_zero = (lhs_zero & rhs_zero) | (lhs_one & rhs_one);
    int known_one = (lhs_zero & rhs_one) | (lhs_one & rhs_zero);

    return std::make_tuple(known_zero, known_one);
}

// Gemini
std::tuple<int, int> abstract_xor(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    int lhs_zeros = std::get(lhs);
    int lhs_ones = std::get(lhs);
    int rhs_zeros = std::get(rhs);
    int rhs_ones = std::get(rhs);

    int result_zeros = 0;
    int result_ones = 0;

    if (lhs_zeros > 0 && rhs_zeros > 0) {
        result_zeros = lhs_zeros + rhs_zeros;
    }

    if (lhs_ones > 0 && rhs_ones > 0) {
        result_zeros = lhs_ones + rhs_ones;
    }

    return std::make_tuple(result_zeros, result_ones);
}

// ChatGPT
std::tuple<int, int> abstract_xor(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    int lhs_known_zeros = std::get<0>(lhs);
    int lhs_known_ones = std::get<1>(lhs);
    int rhs_known_zeros = std::get<0>(rhs);
    int rhs_known_ones = std::get<1>(rhs);

    int known_zeros = (lhs_known_zeros & rhs_known_zeros) | (lhs_known_ones & rhs_known_ones);
    int known_ones = (lhs_known_zeros & rhs_known_ones) | (lhs_known_ones & rhs_known_zeros);

    return {known_zeros, known_ones};
}
```

### `udiv`

```cpp
// DeepSeek
std::tuple<int, int> abstract_udiv(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    int lhs_zeros = std::get<0>(lhs);
    int lhs_ones = std::get<1>(lhs);
    int rhs_zeros = std::get<0>(rhs);
    int rhs_ones = std::get<1>(rhs);

    // If divisor is zero, result is undefined
    if (rhs_ones == 0) {
        return std::make_tuple(0, 0);
    }

    // If divisor is 1, result is the dividend
    if (rhs_zeros == 0 && rhs_ones == 1) {
        return lhs;
    }

    // Otherwise, result is unknown
    return std::make_tuple(0, 0);
}

// DeepSeek (reasoning mode)
std::tuple<int, int> abstract_udiv(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    unsigned l_known0 = std::get<0>(lhs) & 0xFFFFFFFF;
    unsigned l_known1 = std::get<1>(lhs) & 0xFFFFFFFF;
    unsigned l_unknown = ~(l_known0 | l_known1) & 0xFFFFFFFF;
    unsigned l_min = l_known1;
    unsigned l_max = l_known1 | l_unknown;

    unsigned r_known0 = std::get<0>(rhs) & 0xFFFFFFFF;
    unsigned r_known1 = std::get<1>(rhs) & 0xFFFFFFFF;
    unsigned r_unknown = ~(r_known0 | r_known1) & 0xFFFFFFFF;
    unsigned r_min = r_known1;
    unsigned r_max = r_known1 | r_unknown;

    if (r_min == 0) {
        return std::make_tuple(0, 0);
    }

    unsigned min_result = l_min / r_max;
    unsigned max_result = l_max / r_min;
    unsigned xor_result = min_result ^ max_result;

    if (xor_result == 0) {
        unsigned known1 = min_result;
        unsigned known0 = ~known1 & 0xFFFFFFFF;
        return std::make_tuple(known0, known1);
    } else {
        int highest_bit = 31 - __builtin_clz(xor_result);
        unsigned mask_common = (0xFFFFFFFF << (highest_bit + 1)) & 0xFFFFFFFF;
        unsigned common = min_result & mask_common;
        unsigned known1 = common;
        unsigned known0 = (mask_common & ~common) & 0xFFFFFFFF;
        return std::make_tuple(known0, known1);
    }
}

// Gemini
std::tuple<int, int> abstract_udiv(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
  int lhs_zeros = std::get(lhs);
  int lhs_ones = std::get(lhs);
  int rhs_zeros = std::get(rhs);
  int rhs_ones = std::get(rhs);

  int out_zeros = 0;
  int out_ones = 0;

  if (rhs_ones > 0) {
    out_zeros = lhs_zeros - rhs_ones;
    out_zeros = std::max(out_zeros, 0);
  }

  if (lhs_ones > 0 && rhs_zeros == 0) {
    out_ones = lhs_ones;
  }

  return std::make_tuple(out_zeros, out_ones);
}

// Gemini Reasoning
std::tuple<int, int> abstract_udiv(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
  return std::make_tuple(0, 0);
}

// ChatGPT
std::tuple<int, int> abstract_udiv(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    int lhs_known_zeros = std::get<0>(lhs);
    int lhs_known_ones = std::get<1>(lhs);
    int rhs_known_zeros = std::get<0>(rhs);
    int rhs_known_ones = std::get<1>(rhs);

    if (rhs_known_ones == 0) {
        return {0, 0}; // Division by zero or unknown divisor leads to unknown result.
    }

    int result_known_zeros = lhs_known_zeros; // Conservative approximation
    int result_known_ones = 0;

    return {result_known_zeros, result_known_ones};
}

// ChatGPT (Reasoning Mode)
std::tuple<int, int> abstract_udiv(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
    const uint32_t MASK_ALL = 0xFFFFFFFF;
    uint32_t lhsKnownZeros = std::get<0>(lhs);
    uint32_t lhsKnownOnes  = std::get<1>(lhs);
    uint32_t rhsKnownZeros = std::get<0>(rhs);
    uint32_t rhsKnownOnes  = std::get<1>(rhs);

    uint32_t lhsUnknown = MASK_ALL & ~(lhsKnownZeros | lhsKnownOnes);
    uint32_t rhsUnknown = MASK_ALL & ~(rhsKnownZeros | rhsKnownOnes);

    uint32_t lhsMin = lhsKnownOnes;
    uint32_t lhsMax = lhsKnownOnes | lhsUnknown;
    uint32_t rhsMin = rhsKnownOnes;
    uint32_t rhsMax = rhsKnownOnes | rhsUnknown;

    // If divisor may be zero, return top (no known bits)
    if (rhsMin == 0)
        return std::make_tuple(0, 0);

    uint32_t qMin = lhsMin / rhsMax;
    uint32_t qMax = lhsMax / rhsMin;

    uint32_t diff = qMin ^ qMax;
    uint32_t mask;
    if (diff == 0)
        mask = MASK_ALL;
    else {
        uint32_t highestDiff = 32 - __builtin_clz(diff);
        mask = ~((1u << highestDiff) - 1);
    }
    uint32_t common = qMin & mask;
    uint32_t knownOnes = common;
    uint32_t knownZeros = (~common) & mask;

    return std::make_tuple((int)knownZeros, (int)knownOnes);
}
```

## Test Harness:

```cpp
#include <llvm/ADT/APInt.h>
#include <tuple>
#include <vector>
using llvm::APInt;

std::vector<APInt> llm_wrapper(std::vector<APInt> arg0,
                               std::vector<APInt> arg1) {
  auto lhs = std::tuple(static_cast<int>(arg0[0].getZExtValue()),
                        static_cast<int>(arg0[1].getZExtValue()));
  auto rhs = std::tuple(static_cast<int>(arg1[0].getZExtValue()),
                        static_cast<int>(arg1[1].getZExtValue()));
  auto res = shl(lhs, rhs);

  APInt res_0 = APInt(8, static_cast<uint64_t>(std::get<0>(res)));
  APInt res_1 = APInt(8, static_cast<uint64_t>(std::get<1>(res)));

  return {res_0, res_1};
}
```

## Code Analysis

**TODO**
