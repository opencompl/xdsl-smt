#include <cmath>
#include <cstdint>
#include <cstdio>
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
  AbstVal(Domain d, llvm::APInt v) : domain(d), v({~v, v}) {
    assert(d == KNOWN_BITS && "constructor doesn't work for other domains\n");
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
    assert(domain == KNOWN_BITS &&
           "intersectWith is only implimented for KnownBits");
    assert(domain == rhs.domain && "rhs domain must match lhs domain");

    return AbstVal(domain, {zero() & rhs.zero(), one() & rhs.one()});
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
