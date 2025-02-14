#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

#include <llvm/ADT/APInt.h>

enum Domain {
  KNOWN_BITS,
  CONSTANT_RANGE,
};

class AbstVal {
public:
  Domain domain;
  std::vector<llvm::APInt> v;

  AbstVal(Domain d, std::vector<llvm::APInt> v) : domain(d), v(v) {}

  static AbstVal top(Domain d, unsigned int bitwidth) {
    assert((d == KNOWN_BITS || d == CONSTANT_RANGE) &&
           "constructor not impl'd for other domains yet");

    if (d == CONSTANT_RANGE) {
      auto max = llvm::APInt::getMaxValue(bitwidth);
      return AbstVal(d, {max, max});
    } else if (d == KNOWN_BITS) {
      auto min = llvm::APInt::getMinValue(bitwidth);
      return AbstVal(d, {min, min});
    }

    std::unreachable();
  }

  static AbstVal bottom(Domain d, unsigned int bitwidth) {
    assert((d == KNOWN_BITS || d == CONSTANT_RANGE) &&
           "constructor not impl'd for other domains yet\n");
    if (d == CONSTANT_RANGE) {
      auto min = llvm::APInt::getMinValue(bitwidth);
      return AbstVal(d, {min, min});
    } else if (d == KNOWN_BITS) {
      // calling this bottom since known 0's and 1's conflict
      // so no concrete values can be extracted from this
      auto max = llvm::APInt::getMaxValue(bitwidth);
      return AbstVal(d, {max, max});
    }

    std::unreachable();
  }

  static AbstVal fromUnion(Domain d, unsigned int bitwidth,
                           const std::vector<AbstVal> &v) {
    assert((d == KNOWN_BITS || d == CONSTANT_RANGE) &&
           "constructor not impl'd for other domains yet\n");
    return std::accumulate(v.begin(), v.end(), AbstVal::top(d, bitwidth),
                           [](const AbstVal &lhs, const AbstVal &rhs) {
                             return lhs.unionWith(rhs);
                           });
  }

  static AbstVal fromIntersection(Domain d, unsigned int bitwidth,
                                  const std::vector<AbstVal> &v) {
    assert((d == KNOWN_BITS || d == CONSTANT_RANGE) &&
           "constructor not impl'd for other domains yet");
    if(v.size() == 0) return AbstVal::bottom(d, bitwidth);
    return std::accumulate(v.begin(), v.end(), v[0],
                           [](const AbstVal &lhs, const AbstVal &rhs) {
                             return lhs.intersectWith(rhs);
                           });
  }

  static AbstVal fromConcrete(Domain d, llvm::APInt v) {
    assert((d == KNOWN_BITS || d == CONSTANT_RANGE) &&
           "constructor not impl'd for other domains yet");
    if (d == KNOWN_BITS) {
      return AbstVal(d, {~v, v});
    } else if (d == CONSTANT_RANGE) {
      return AbstVal(d, {v, v + 1});
    }

    std::unreachable();
  }

  bool isSuperset(const AbstVal &rhs) const {
    return this->unionWith(rhs) == rhs;
  }

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

  // TODO fix this
  // add asserts etc
  bool isUpperWrapped() const { return lower().ugt(upper()); }

  // TODO fix this
  // add asserts etc
  static AbstVal getPreferredRange(const AbstVal &lhs, const AbstVal &rhs) {
    if (lhs.isSizeStrictlySmallerThan(rhs))
      return lhs;
    return rhs;
  }

  // TODO fix this too
  // add asserts etc
  bool isSizeStrictlySmallerThan(const AbstVal &rhs) const {
    if (isFullSet())
      return false;
    if (rhs.isFullSet())
      return true;
    return (upper() - lower()).ult(rhs.upper() - rhs.lower());
  }

  AbstVal unionWith(const AbstVal &rhs) const {
    assert(domain == rhs.domain && "lhs and rhs domains must match");
    assert((domain == KNOWN_BITS || domain == CONSTANT_RANGE) &&
           "function not impl'd for other domains yet");

    if (domain == KNOWN_BITS) {
      return AbstVal(KNOWN_BITS, {zero() | rhs.zero(), one() | rhs.one()});
    } else if (domain == CONSTANT_RANGE) {
      // pretty much just ripped all this code from
      // llvm/lib/IR/ConstantRange.cpp
      // the fact that the ranges are wrapped makes this a huge PITA
      if (isFullSet() || rhs.isEmptySet())
        return *this;
      if (rhs.isFullSet() || isEmptySet())
        return rhs;

      if (!isUpperWrapped() && rhs.isUpperWrapped())
        return rhs.unionWith(*this);

      if (!isUpperWrapped() && !rhs.isUpperWrapped()) {
        if (rhs.upper().ult(lower()) || upper().ult(rhs.lower()))
          // TODO probs just use unsigned for the prefered constant range
          return getPreferredRange(
              AbstVal(CONSTANT_RANGE, {lower(), rhs.upper()}),
              AbstVal(CONSTANT_RANGE, {rhs.lower(), upper()}));

        llvm::APInt L = rhs.lower().ult(lower()) ? rhs.lower() : lower();
        llvm::APInt U =
            (rhs.upper() - 1).ugt(upper() - 1) ? rhs.upper() : upper();

        if (L.isZero() && U.isZero())
          return top(CONSTANT_RANGE, lower().getBitWidth());

        AbstVal(CONSTANT_RANGE, {std::move(L), std::move(U)});
      }

      if (!rhs.isUpperWrapped()) {
        if (rhs.upper().ule(upper()) || rhs.lower().uge(lower()))
          return *this;

        if (rhs.lower().ule(upper()) && lower().ule(rhs.upper()))
          return top(CONSTANT_RANGE, lower().getBitWidth());

        if (upper().ult(rhs.lower()) && rhs.upper().ult(lower()))
          return getPreferredRange(
              AbstVal(CONSTANT_RANGE, {lower(), rhs.upper()}),
              AbstVal(CONSTANT_RANGE, {rhs.lower(), upper()}));

        if (upper().ult(rhs.lower()) && lower().ule(rhs.upper()))
          return AbstVal(CONSTANT_RANGE, {rhs.lower(), upper()});

        assert(rhs.lower().ule(upper()) && rhs.upper().ult(lower()) &&
               "ConstantRange::unionWith missed a case with one range "
               "wrapped");
        return AbstVal(CONSTANT_RANGE, {lower(), rhs.upper()});
      }

      if (rhs.lower().ule(upper()) || lower().ule(rhs.upper()))
        return top(CONSTANT_RANGE, lower().getBitWidth());

      llvm::APInt L = rhs.lower().ult(lower()) ? rhs.lower() : lower();
      llvm::APInt U = rhs.upper().ugt(upper()) ? rhs.upper() : upper();

      return AbstVal(CONSTANT_RANGE, {std::move(L), std::move(U)});
    }

    std::unreachable();
  }

  void printAbstRange() const {
    if (domain == KNOWN_BITS) {
      for (uint32_t i = zero().getBitWidth() - 1; i >= 0; --i) {
        const char bit = one()[i] ? '1' : zero()[i] ? '0' : '?';
        printf("%c", bit);
      }
      if (isConstant())
        printf(" const %lu", getConstant().getZExtValue());
      if (isUnknown())
        printf(" (top)");
      printf("\n");
    } else if (domain == CONSTANT_RANGE) {
      printf("[%ld, %ld)", getLower().getZExtValue(),
             getUpper().getZExtValue());
      if (isFullSet())
        printf(" (top)");
      if (isEmptySet())
        printf(" (bottom)");
      printf("\n");
    } else {
      printf("unknown domain\n");
    }
  }

  // TODO return a generic container based on what the caller asks for
  // TODO there's a faster way to this but this works for now
  // TODO should this return an APInt??
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
      std::vector<uint8_t> ret;

      if (isFullSet()) {
        const llvm::APInt min =
            llvm::APInt::getMinValue(getLower().getBitWidth());
        const llvm::APInt max =
            llvm::APInt::getMaxValue(getLower().getBitWidth());

        for (llvm::APInt i = min; i != max; ++i)
          ret.push_back(static_cast<uint8_t>(i.getZExtValue()));

        ret.push_back(static_cast<uint8_t>(max.getZExtValue()));

        return ret;
      }

      for (llvm::APInt i = getLower(); i != getUpper(); ++i)
        ret.push_back(static_cast<uint8_t>(i.getZExtValue()));

      return ret;
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

  AbstVal intersectWith(const AbstVal &rhs) const {
    assert(domain == rhs.domain && "rhs and lhs domain must match");
    assert((domain == KNOWN_BITS || domain == CONSTANT_RANGE) &&
           "function not impl'd for this domain");

    if (domain == KNOWN_BITS) {
      return AbstVal(domain, {zero() & rhs.zero(), one() & rhs.one()});
    } else if (domain == CONSTANT_RANGE) {
      // also pretty much just ripped out of
      // llvm/lib/IR/ConstantRange.cpp
      if (isEmptySet() || rhs.isFullSet())
        return *this;
      if (rhs.isEmptySet() || isFullSet())
        return rhs;

      if (!isUpperWrapped() && rhs.isUpperWrapped())
        return rhs.intersectWith(*this);

      if (!isUpperWrapped() && !rhs.isUpperWrapped()) {
        if (lower().ult(rhs.lower())) {
          if (upper().ule(rhs.lower()))
            return bottom(CONSTANT_RANGE, lower().getBitWidth());

          if (upper().ult(rhs.upper()))
            return AbstVal(CONSTANT_RANGE, {rhs.lower(), upper()});

          return rhs;
        }
        if (upper().ult(rhs.upper()))
          return *this;

        if (lower().ult(rhs.upper()))
          return AbstVal(CONSTANT_RANGE, {lower(), rhs.upper()});

        return bottom(CONSTANT_RANGE, lower().getBitWidth());
      }

      if (isUpperWrapped() && !rhs.isUpperWrapped()) {
        if (rhs.lower().ult(upper())) {
          if (rhs.upper().ult(upper()))
            return rhs;

          if (rhs.upper().ule(lower()))
            return AbstVal(CONSTANT_RANGE, {rhs.lower(), upper()});

          return getPreferredRange(*this, rhs);
        }
        if (rhs.lower().ult(lower())) {
          if (rhs.upper().ule(lower()))
            return bottom(CONSTANT_RANGE, lower().getBitWidth());

          return AbstVal(CONSTANT_RANGE, {lower(), rhs.upper()});
        }

        return rhs;
      }

      if (rhs.upper().ult(upper())) {
        if (rhs.lower().ult(upper()))
          return getPreferredRange(*this, rhs);

        if (rhs.lower().ult(lower()))
          return AbstVal(CONSTANT_RANGE, {lower(), rhs.upper()});

        return rhs;
      }
      if (rhs.upper().ule(lower())) {
        if (rhs.lower().ult(lower()))
          return *this;

        return AbstVal(CONSTANT_RANGE, {rhs.lower(), upper()});
      }

      return getPreferredRange(*this, rhs);
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

  unsigned getBitWidth() const {
    assert(domain == KNOWN_BITS &&
           "getBitWidth is only applicable to the KnownBits domain");
    assert(zero().getBitWidth() == one().getBitWidth() &&
           "Zero and One should have the same width!");
    return zero().getBitWidth();
  }

  bool isConstant() const {
    assert(domain == KNOWN_BITS &&
           "isConstant is only applicable to the KnownBits domain");

    return zero().popcount() + one().popcount() == getBitWidth();
  }

  bool isUnknown() const {
    assert(domain == KNOWN_BITS &&
           "isUnknown is only applicable to the KnownBits domain");
    return zero().isZero() && one().isZero();
  }

  const llvm::APInt getConstant() const {
    assert(domain == KNOWN_BITS &&
           "getConstant is only applicable to the KnownBits domain");
    assert(isConstant() && "Can only get value when all bits are known");
    return one();
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

  llvm::APInt getLower() const {
    assert(domain == CONSTANT_RANGE &&
           "getLower is only applicable to the KnownBits domain");
    return lower();
  }

  llvm::APInt getUpper() const {
    assert(domain == CONSTANT_RANGE &&
           "getUpper is only applicable to the KnownBits domain");
    return upper();
  }

  bool isEmptySet() const {
    assert(domain == CONSTANT_RANGE &&
           "isEmptySet is only applicable to the KnownBits domain");
    return lower() == upper() && lower().isMinValue();
  }

  bool isFullSet() const {
    assert(domain == CONSTANT_RANGE &&
           "isFullSet is only applicable to the KnownBits domain");
    return lower() == upper() && lower().isMaxValue();
  }
};
