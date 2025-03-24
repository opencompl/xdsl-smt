#ifndef AbstVal_H
#define AbstVal_H

#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "APInt.h"

template <typename Domain, unsigned int N> class AbstVal {
protected:
  explicit AbstVal(const Vec<N> &x)
      : v(x.v, x.v + x.getN()), bw(x[0].getBitWidth()) {}

public:
  typedef Vec<N> (*XferFn)(Vec<N>, Vec<N>);

  std::vector<A::APInt> v;
  unsigned int bw;

  // static ctors
  static const Domain bottom(unsigned int bw) { return Domain::bottom(bw); }
  static const Domain top(unsigned int bw) { return Domain::top(bw); }
  static const std::vector<Domain> enumVals() { return Domain::enumVals(); }

  static const Domain fromConcrete(const A::APInt &x) {
    return Domain::fromConcrete(x);
  }

  static const Domain joinAll(const std::vector<Domain> &v, unsigned int bw) {
    return std::accumulate(
        v.begin(), v.end(), bottom(bw),
        [](const Domain &lhs, const Domain &rhs) { return lhs.join(rhs); });
  }

  static const Domain meetAll(const std::vector<Domain> &v, unsigned int bw) {
    return std::accumulate(
        v.begin(), v.end(), top(bw),
        [](const Domain &lhs, const Domain &rhs) { return lhs.meet(rhs); });
  }

  // normal methods
  bool isTop() const { return *this == top(bw); }
  bool isBottom() const { return *this == bottom(bw); }
  bool isSuperset(const Domain &rhs) const { return meet(rhs) == rhs; }
  unsigned int distance(const Domain &rhs) const {
    return (v[0] ^ rhs.v[0]).popcount() + (v[1] ^ rhs.v[1]).popcount();
  }

  bool operator==(const AbstVal &rhs) const {
    for (unsigned int i = 0; i < N; ++i)
      if (v[i] != rhs.v[i])
        return false;

    return true;
  };

  // methods delegated to derived class
  bool isConstant() const {
    return static_cast<const Domain *>(this)->isConstant();
  };
  const A::APInt getConstant() const {
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

class KnownBits : public AbstVal<KnownBits, 2> {
private:
  A::APInt zero() const { return v[0]; }
  A::APInt one() const { return v[1]; }
  bool hasConflict() const { return zero().intersects(one()); }

public:
  explicit KnownBits(const Vec<2> &vC) : AbstVal<KnownBits, 2>(vC) {}

  const std::string display() const {
    if (KnownBits::isBottom()) {
      return "(bottom)";
    }

    std::stringstream ss;

    for (unsigned int i = bw; i > 0; --i)
      ss << (one()[i - 1] ? '1' : zero()[i - 1] ? '0' : '?');

    if (isConstant())
      ss << getConstant().getZExtValue();

    if (KnownBits::isTop())
      ss << " (top)";

    return ss.str();
  }

  bool isConstant() const { return zero().popcount() + one().popcount() == bw; }
  const A::APInt getConstant() const { return zero(); }

  const KnownBits meet(const KnownBits &rhs) const {
    return KnownBits({zero() | rhs.zero(), one() | rhs.one()});
  }

  const KnownBits join(const KnownBits &rhs) const {
    return KnownBits({zero() & rhs.zero(), one() & rhs.one()});
  }

  const std::vector<unsigned int> toConcrete() const {
    std::vector<unsigned int> ret;
    const unsigned int z = static_cast<unsigned int>(zero().getZExtValue());
    const unsigned int o = static_cast<unsigned int>(one().getZExtValue());
    const unsigned int min =
        static_cast<unsigned int>(A::APInt::getZero(bw).getZExtValue());
    const unsigned int max =
        static_cast<unsigned int>(A::APInt::getMaxValue(bw).getZExtValue());

    for (unsigned int i = min; i <= max; ++i)
      if ((z & i) == 0 && (o & ~i) == 0)
        ret.push_back(i);

    return ret;
  }

  static KnownBits fromConcrete(const A::APInt &x) {
    return KnownBits({~x, x});
  }

  static KnownBits bottom(unsigned int bw) {
    A::APInt max = A::APInt::getMaxValue(bw);
    return KnownBits({max, max});
  }

  static KnownBits top(unsigned int bw) {
    A::APInt min = A::APInt::getMinValue(bw);
    return KnownBits({min, min});
  }

  static std::vector<KnownBits> const enumVals(unsigned int bw) {
    const unsigned int max =
        static_cast<unsigned int>(A::APInt::getMaxValue(bw).getZExtValue());
    A::APInt zero = A::APInt(bw, 0);
    A::APInt one = A::APInt(bw, 0);
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

class ConstantRange : public AbstVal<ConstantRange, 2> {
private:
  A::APInt lower() const { return v[0]; }
  A::APInt upper() const { return v[1]; }

public:
  explicit ConstantRange(const Vec<2> &vC) : AbstVal<ConstantRange, 2>(vC) {}

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
  const A::APInt getConstant() const { return lower(); }

  const ConstantRange meet(const ConstantRange &rhs) const {
    A::APInt l = rhs.lower().ugt(lower()) ? rhs.lower() : lower();
    A::APInt u = rhs.upper().ult(upper()) ? rhs.upper() : upper();
    if (l.ugt(u))
      return bottom(bw);
    return ConstantRange({std::move(l), std::move(u)});
  }

  const ConstantRange join(const ConstantRange &rhs) const {
    const A::APInt l = rhs.lower().ult(lower()) ? rhs.lower() : lower();
    const A::APInt u = rhs.upper().ugt(upper()) ? rhs.upper() : upper();
    return ConstantRange({std::move(l), std::move(u)});
  }

  const std::vector<unsigned int> toConcrete() const {
    unsigned int l = static_cast<unsigned int>(lower().getZExtValue());
    unsigned int u = static_cast<unsigned int>(upper().getZExtValue() + 1);

    if (l > u)
      return {};

    std::vector<unsigned int> ret(u - l);
    std::iota(ret.begin(), ret.end(), l);
    return ret;
  }

  static ConstantRange fromConcrete(const A::APInt &x) {
    return ConstantRange({x, x});
  }

  static ConstantRange bottom(unsigned int bw) {
    A::APInt min = A::APInt::getMinValue(bw);
    A::APInt max = A::APInt::getMaxValue(bw);
    return ConstantRange({max, min});
  }

  static ConstantRange top(unsigned int bw) {
    A::APInt min = A::APInt::getMinValue(bw);
    A::APInt max = A::APInt::getMaxValue(bw);
    return ConstantRange({min, max});
  }

  static std::vector<ConstantRange> const enumVals(unsigned int bw) {
    const unsigned int min =
        static_cast<unsigned int>(A::APInt::getMinValue(bw).getZExtValue());
    const unsigned int max =
        static_cast<unsigned int>(A::APInt::getMaxValue(bw).getZExtValue());
    A::APInt l = A::APInt(bw, 0);
    A::APInt u = A::APInt(bw, 0);
    std::vector<ConstantRange> ret = {top(bw)};

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

#endif
