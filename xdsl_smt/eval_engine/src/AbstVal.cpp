#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <ranges>
#include <utility>
#include <vector>

#include <llvm/ADT/APInt.h>

// NOTES:
// alpha (abstraction function) AbstactValue -> Set[ConcreteValue]
// gamma (concretization function) Set[ConcreteValue] -> AbstactValue
//
// meet (greatest lower bound)
//   set[AbstractValue] -> AbstractValue
//   the meet of the empty set should return top
//
// join (least upper bound)
//   set[AbstractValue] -> AbstractValue
//   the join of the empty set should return bottom

// big TODO:
// this would be WAY easier to maintain/less bug prone if I used interfaces
// C++ Concepts seem to provide a reasonable implimentaion for these

enum Domain {
  KNOWN_BITS,
  CONSTANT_RANGE,
};

class AbstVal {
public:
  Domain domain;
  std::vector<llvm::APInt> v;
  unsigned int bitwidth;

  // TODO probs make this private and only accesible from some friend function,
  // that indicates that you should not be constructing an AbstVal this way
  AbstVal(Domain d, std::vector<llvm::APInt> v, unsigned int bw)
      : domain(d), v(v), bitwidth(bw) {}

  static AbstVal top(Domain d, unsigned int bitwidth) {
    assert((d == KNOWN_BITS || d == CONSTANT_RANGE) &&
           "constructor not impl'd for other domains yet");
    auto max = llvm::APInt::getMaxValue(bitwidth);
    auto min = llvm::APInt::getMinValue(bitwidth);

    if (d == CONSTANT_RANGE)
      return AbstVal(d, {min, max}, bitwidth);
    else if (d == KNOWN_BITS)
      return AbstVal(d, {min, min}, bitwidth);

    std::unreachable();
  }

  static AbstVal bottom(Domain d, unsigned int bitwidth) {
    assert((d == KNOWN_BITS || d == CONSTANT_RANGE) &&
           "constructor not impl'd for other domains yet\n");
    auto min = llvm::APInt::getMinValue(bitwidth);
    auto max = llvm::APInt::getMaxValue(bitwidth);

    // both KB and CR bottoms were picked bc they have invalid bit
    // representaions, and they have the opposite bit pattern of top
    if (d == CONSTANT_RANGE)
      return AbstVal(d, {max, min}, bitwidth);
    else if (d == KNOWN_BITS)
      return AbstVal(d, {max, max}, bitwidth);

    std::unreachable();
  }

  static AbstVal joinAll(Domain d, unsigned int bitwidth,
                         const std::vector<AbstVal> &v) {
    assert((d == KNOWN_BITS || d == CONSTANT_RANGE) &&
           "constructor not impl'd for other domains yet\n");
    return std::accumulate(
        v.begin(), v.end(), AbstVal::bottom(d, bitwidth),
        [](const AbstVal &lhs, const AbstVal &rhs) { return lhs.join(rhs); });
  }

  static AbstVal meetAll(Domain d, unsigned int bitwidth,
                         const std::vector<AbstVal> &v) {
    assert((d == KNOWN_BITS || d == CONSTANT_RANGE) &&
           "constructor not impl'd for other domains yet");
    if (d == KNOWN_BITS) {
      return std::accumulate(
          v.begin(), v.end(), AbstVal::top(d, bitwidth),
          [](const AbstVal &lhs, const AbstVal &rhs) { return lhs.meet(rhs); });
    } else if (d == CONSTANT_RANGE) {
      if (v.size() == 0)
        return AbstVal::top(d, bitwidth);

      auto [l, u] = std::minmax_element(v.begin(), v.end(),
                                        [](const AbstVal &a, const AbstVal &b) {
                                          return a.v[0].ult(b.v[0]);
                                        });

      return AbstVal(d, {l->v[0], u->v[0]}, bitwidth);
    }

    std::unreachable();
  }

  // also known as alpha
  static AbstVal fromConcrete(Domain d, llvm::APInt v) {
    assert((d == KNOWN_BITS || d == CONSTANT_RANGE) &&
           "constructor not impl'd for other domains yet");
    if (d == KNOWN_BITS) {
      return AbstVal(d, {~v, v}, v.getBitWidth());
    } else if (d == CONSTANT_RANGE) {
      return AbstVal(d, {v, v}, v.getBitWidth());
    }

    std::unreachable();
  }

  // TODO should be purged if we move ta concepts
  bool isSuperset(const AbstVal &rhs) const { return this->meet(rhs) == rhs; }

  // TODO should be purged if we move ta concepts
  unsigned int distance(const AbstVal &rhs) const {
    return (v[0] ^ rhs.v[0]).popcount() + (v[1] ^ rhs.v[1]).popcount();
  }

  bool operator==(const AbstVal &rhs) const {
    if (domain != rhs.domain)
      return false;
    if (v.size() != rhs.v.size())
      return false;

    for (uint32_t i = 0; i < v.size(); ++i) {
      if (v[i] != rhs.v[i])
        return false;
    }

    return true;
  }

  AbstVal join(const AbstVal &rhs) const {
    assert(domain == rhs.domain && "lhs and rhs domains must match");
    assert((domain == KNOWN_BITS || domain == CONSTANT_RANGE) &&
           "function not impl'd for other domains yet");

    if (domain == KNOWN_BITS) {
      return AbstVal(KNOWN_BITS, {zero() & rhs.zero(), one() & rhs.one()},
                     bitwidth);
    } else if (domain == CONSTANT_RANGE) {
      llvm::APInt L = rhs.lower().ult(lower()) ? rhs.lower() : lower();
      llvm::APInt U = rhs.upper().ugt(upper()) ? rhs.upper() : upper();
      return AbstVal(CONSTANT_RANGE, {std::move(L), std::move(U)}, bitwidth);
    }

    std::unreachable();
  }

  void printAbstRange() const {
    if (domain == KNOWN_BITS) {
      if (isBottom()) {
        printf("(bottom)\n");
        return;
      }
      for (uint32_t i = zero().getBitWidth(); i > 0; --i) {
        const char bit = one()[i - 1] ? '1' : zero()[i - 1] ? '0' : '?';
        printf("%c", bit);
      }
      if (isConstant())
        printf(" const %lu", getConstant().getZExtValue());
      if (isTop())
        printf(" (top)");
      printf("\n");
    } else if (domain == CONSTANT_RANGE) {
      if (isBottom()) {
        printf("(bottom)\n");
        return;
      }
      printf("[%ld, %ld]", lower().getZExtValue(), upper().getZExtValue());
      if (isTop())
        printf(" (top)");
      printf("\n");
    } else {
      fprintf(stderr, "unknown domain\n");
    }
  }

  // TODO return a generic container based on what the caller asks for
  // TODO there's a faster way to this but this works for now
  // TODO should this return an APInt??
  // also known as alpha
  std::vector<uint8_t> const toConcrete() const {
    if (domain == KNOWN_BITS) {
      std::vector<uint8_t> ret;
      const llvm::APInt min = llvm::APInt::getZero(zero().getBitWidth());
      const llvm::APInt max = llvm::APInt::getMaxValue(zero().getBitWidth());

      for (auto i = min;; ++i) {

        if (!zero().intersects(i) && !one().intersects(~i))
          ret.push_back(static_cast<uint8_t>(i.getZExtValue()));

        if (i == max)
          break;
      }

      return ret;

    } else if (domain == CONSTANT_RANGE) {
      uint8_t l = static_cast<uint8_t>(lower().getZExtValue());
      uint8_t u = static_cast<uint8_t>(upper().getZExtValue() + 1);

      if (l > u)
        return {};

      return std::views::iota(l, u) | std::ranges::to<std::vector>();
    } else {
      printf("unknown domain\n");
    }

    return {};
  }

  // public kb stuff
  bool hasConflict() const {
    assert(domain == KNOWN_BITS &&
           "hasConflict is only applicable to the KnownBits domain");
    return zero().intersects(one());
  }

  AbstVal meet(const AbstVal &rhs) const {
    assert(domain == rhs.domain && "rhs and lhs domain must match");
    assert((domain == KNOWN_BITS || domain == CONSTANT_RANGE) &&
           "function not impl'd for this domain");

    if (domain == KNOWN_BITS) {
      return AbstVal(domain, {zero() | rhs.zero(), one() | rhs.one()},
                     bitwidth);
    } else if (domain == CONSTANT_RANGE) {
      llvm::APInt l = rhs.lower().ugt(lower()) ? rhs.lower() : lower();
      llvm::APInt u = rhs.upper().ult(upper()) ? rhs.upper() : upper();
      if (l.ugt(u))
        return bottom(CONSTANT_RANGE, l.getBitWidth());
      return AbstVal(CONSTANT_RANGE, {std::move(l), std::move(u)}, bitwidth);
    }

    std::unreachable();
  }

private:
  // kb stuff
  llvm::APInt const zero() const {
    assert(domain == KNOWN_BITS &&
           "zero is only applicable to the KnownBits domain");
    return v[0];
  }

  llvm::APInt const one() const {
    assert(domain == KNOWN_BITS &&
           "one is only applicable to the KnownBits domain");
    return v[1];
  }

  const llvm::APInt getConstant() const {
    assert((domain == KNOWN_BITS || domain == CONSTANT_RANGE) &&
           "isConstant is only applicable to the KnownBits domain");
    assert(isConstant() && "Can only get value when all bits are known");
    if (domain == KNOWN_BITS)
      return one();
    if (domain == KNOWN_BITS)
      return upper();

    std::unreachable();
  }

  // cr stuff
  llvm::APInt const lower() const {
    assert(domain == CONSTANT_RANGE &&
           "lower is only applicable to the KnownBits domain");
    return v[0];
  }

  llvm::APInt const upper() const {
    assert(domain == CONSTANT_RANGE &&
           "upper is only applicable to the KnownBits domain");
    return v[1];
  }

  bool isConstant() const {
    assert((domain == KNOWN_BITS || domain == CONSTANT_RANGE) &&
           "isConstant is only applicable to the KnownBits domain");

    if (domain == KNOWN_BITS)
      return zero().popcount() + one().popcount() == bitwidth;
    if (domain == CONSTANT_RANGE)
      return lower() == upper();

    std::unreachable();
  }

  bool isBottom() const { return *this == bottom(domain, bitwidth); }
  bool isTop() const { return *this == top(domain, bitwidth); }
};
