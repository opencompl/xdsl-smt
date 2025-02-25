#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <llvm/ADT/APInt.h>

template <typename Domain, unsigned char N> class AbstVal {
protected:
  explicit AbstVal(const std::vector<llvm::APInt> &v) : v(v) {}

public:
  std::vector<llvm::APInt> v;

  // static ctors
  static const Domain bottom() { return Domain::bottom(); }
  static const Domain top() { return Domain::top(); }
  static const std::vector<Domain> enumVals() { return Domain::enumVals(); }

  static const Domain fromConcrete(const llvm::APInt &x) {
    return Domain::fromConcrete(x);
  }

  static const Domain joinAll(const std::vector<Domain> &v) {
    return std::accumulate(
        v.begin(), v.end(), bottom(),
        [](const Domain &lhs, const Domain &rhs) { return lhs.join(rhs); });
  }

  static const Domain meetAll(const std::vector<Domain> &v) {
    return std::accumulate(
        v.begin(), v.end(), top(),
        [](const Domain &lhs, const Domain &rhs) { return lhs.meet(rhs); });
  }

  // normal methods
  unsigned char getBitWidth() const { return N; }
  bool isTop() const { return *this == top(); }
  bool isBottom() const { return *this == bottom(); }
  bool isSuperset(const Domain &rhs) const { return meet(rhs) == rhs; }
  unsigned int distance(const Domain &rhs) const {
    return (v[0] ^ rhs.v[0]).popcount() + (v[1] ^ rhs.v[1]).popcount();
  }

  bool operator==(const AbstVal &rhs) const {
    if (v.size() != rhs.v.size())
      return false;

    for (unsigned long i = 0; i < v.size(); ++i)
      if (v[i] != rhs.v[i])
        return false;

    return true;
  };

  // methods delegated to derived class
  bool isConstant() const {
    return static_cast<const Domain *>(this)->isConstant();
  };
  const llvm::APInt getConstant() const {
    return static_cast<const Domain *>(this)->getConstant();
  };
  const Domain meet(const Domain &rhs) const {
    return static_cast<const Domain *>(this)->meet(rhs);
  };
  const Domain join(const Domain &rhs) const {
    return static_cast<const Domain *>(this)->join(rhs);
  };
  const std::vector<unsigned int> toConcrete() const {
    return static_cast<const Domain *>(this)->toConcrete();
  };
  const std::string display() const {
    return static_cast<Domain>(this)->display();
  };
};

template <unsigned char N> class KnownBits : public AbstVal<KnownBits<N>, N> {
private:
  llvm::APInt zero() const { return this->v[0]; }
  llvm::APInt one() const { return this->v[1]; }
  bool hasConflict() const { return zero().intersects(one()); }

public:
  explicit KnownBits(const std::vector<llvm::APInt> &v)
      : AbstVal<KnownBits<N>, N>(v) {}

  const std::string display() const {
    if (KnownBits<N>::isBottom()) {
      return "(bottom)";
    }

    std::stringstream ss;

    for (unsigned int i = N; i > 0; --i)
      ss << (one()[i - 1] ? '1' : zero()[i - 1] ? '0' : '?');

    if (isConstant())
      ss << getConstant().getZExtValue();

    if (KnownBits<N>::isTop())
      ss << " (top)";

    return ss.str();
  }

  bool isConstant() const { return zero().popcount() + one().popcount() == N; }
  const llvm::APInt getConstant() const { return zero(); }

  const KnownBits meet(const KnownBits &rhs) const {
    return KnownBits({zero() | rhs.zero(), one() | rhs.one()});
  }

  const KnownBits join(const KnownBits &rhs) const {
    return KnownBits({zero() & rhs.zero(), one() & rhs.one()});
  }

  const std::vector<unsigned int> toConcrete() const {
    std::vector<unsigned int> ret;
    const unsigned int z = zero().getZExtValue();
    const unsigned int o = one().getZExtValue();
    const unsigned int min = llvm::APInt::getZero(N).getZExtValue();
    const unsigned int max = llvm::APInt::getMaxValue(N).getZExtValue();

    for (unsigned int i = min; i <= max; ++i)
      if ((z & i) == 0 && (o & ~i) == 0)
        ret.push_back(i);

    return ret;
  }

  static KnownBits fromConcrete(const llvm::APInt &x) {
    return KnownBits({~x, x});
  }

  static KnownBits bottom() {
    llvm::APInt max = llvm::APInt::getMaxValue(N);
    return KnownBits({max, max});
  }

  static KnownBits top() {
    llvm::APInt min = llvm::APInt::getMinValue(N);
    return KnownBits({min, min});
  }

  static std::vector<KnownBits> const enumVals() {
    const unsigned int max = llvm::APInt::getMaxValue(N).getZExtValue();
    llvm::APInt zero = llvm::APInt(N, 0);
    llvm::APInt one = llvm::APInt(N, 0);
    std::vector<KnownBits> ret;
    ret.reserve(max * max);

    for (unsigned int i = 0; i <= max; ++i) {
      unsigned char jmp = i % 2 + 1;
      for (unsigned int j = 0; j <= max; j += jmp) {
        if ((i & j) != 0)
          continue;

        zero = i;
        one = j;
        ret.push_back(KnownBits({zero, one}));
      }
    }

    return ret;
  }
};

template <unsigned char N>
class ConstantRange : public AbstVal<ConstantRange<N>, N> {
private:
  llvm::APInt lower() const { return this->v[0]; }
  llvm::APInt upper() const { return this->v[1]; }

public:
  explicit ConstantRange(const std::vector<llvm::APInt> &v)
      : AbstVal<ConstantRange<N>, N>(v) {}

  const std::string display() const {
    if (ConstantRange::isBottom()) {
      return "(bottom)";
    }

    std::stringstream ss;
    ss << '[' << lower().getZExtValue() << ", " << upper().getZExtValue()
       << ']';

    if (ConstantRange::isTop())
      ss << " (top)";

    return ss.str();
  }

  bool isConstant() const { return lower() == upper(); }
  const llvm::APInt getConstant() const { return lower(); }

  const ConstantRange meet(const ConstantRange &rhs) const {
    llvm::APInt l = rhs.lower().ugt(lower()) ? rhs.lower() : lower();
    llvm::APInt u = rhs.upper().ult(upper()) ? rhs.upper() : upper();
    if (l.ugt(u))
      return bottom();
    return ConstantRange({std::move(l), std::move(u)});
  }

  const ConstantRange join(const ConstantRange &rhs) const {
    const llvm::APInt l = rhs.lower().ult(lower()) ? rhs.lower() : lower();
    const llvm::APInt u = rhs.upper().ugt(upper()) ? rhs.upper() : upper();
    return ConstantRange({std::move(l), std::move(u)});
  }

  const std::vector<unsigned int> toConcrete() const {
    unsigned int l = lower().getZExtValue();
    unsigned int u = upper().getZExtValue() + 1;

    if (l > u)
      return {};

    std::vector<unsigned int> ret(u - l);
    std::iota(ret.begin(), ret.end(), l);
    return ret;
  }

  static ConstantRange fromConcrete(const llvm::APInt &x) {
    return ConstantRange({x, x});
  }

  static ConstantRange bottom() {
    llvm::APInt min = llvm::APInt::getMinValue(N);
    llvm::APInt max = llvm::APInt::getMaxValue(N);
    return ConstantRange({max, min});
  }

  static ConstantRange top() {
    llvm::APInt min = llvm::APInt::getMinValue(N);
    llvm::APInt max = llvm::APInt::getMaxValue(N);
    return ConstantRange({min, max});
  }

  static std::vector<ConstantRange> const enumVals() {
    const unsigned int min = llvm::APInt::getMinValue(N).getZExtValue();
    const unsigned int max = llvm::APInt::getMaxValue(N).getZExtValue();
    llvm::APInt l = llvm::APInt(N, 0);
    llvm::APInt u = llvm::APInt(N, 0);
    std::vector<ConstantRange> ret = {top()};

    for (unsigned int i = min; i <= max; ++i) {
      for (unsigned int j = i; j <= max; ++j) {
        l = i;
        u = j;
        ret.push_back(ConstantRange({l, u}));
      }
    }

    return ret;
  }
};
