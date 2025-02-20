#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <llvm/ADT/APInt.h>

// TODO put this somewhere else
uint64_t makeMask(uint8_t bitwidth) {
  if (bitwidth == 0)
    return 0;
  return (1 << bitwidth) - 1;
}

// TODO maybe also template over bitwidth?
//
// template <typename Domain, int N> class AbstVal {
template <typename D> class AbstVal {
  // TODO make these protected again
  // just doing it this way for synth.cpp
protected:
public:
  // TODO I wish these were const but no such luck
  // const
  std::vector<llvm::APInt> v;
  // const
  unsigned int bitwidth;
  explicit AbstVal(const std::vector<llvm::APInt> &v, unsigned int bw)
      : v(v), bitwidth(bw) {}

public:
  static const D joinAll(unsigned int bw, const std::vector<D> &v) {
    return std::accumulate(
        v.begin(), v.end(), bottom(bw),
        [](const D &lhs, const D &rhs) { return lhs.join(rhs); });
  }

  static const D meetAll(unsigned int bw, const std::vector<D> &v) {
    return std::accumulate(
        v.begin(), v.end(), top(bw),
        [](const D &lhs, const D &rhs) { return lhs.meet(rhs); });
  }

  static const D fromConcrete(const llvm::APInt &x) {
    static_assert(std::is_base_of<AbstVal, D>::value);
    static_assert(std::is_same<decltype(&D::fromConcrete),
                               D (*)(const llvm::APInt &)>::value);

    return D::fromConcrete(x);
  }

  static const D bottom(unsigned int bw) {
    static_assert(std::is_base_of<AbstVal, D>::value);
    static_assert(
        std::is_same<decltype(&D::bottom), D (*)(unsigned int)>::value);

    return D::bottom(bw);
  }

  static const D top(unsigned int bw) {
    static_assert(std::is_base_of<AbstVal, D>::value);
    static_assert(std::is_same<decltype(&D::top), D (*)(unsigned int)>::value);

    return D::top(bw);
  }

  static const std::vector<D> enumVals(unsigned int bw) {
    static_assert(std::is_base_of<AbstVal, D>::value);
    static_assert(
        std::is_same<decltype(&D::enumVals), D (*)(unsigned int)>::value);

    return D::enumVals(bw);
  }

  bool operator==(const AbstVal &rhs) const {
    if (bitwidth != rhs.bitwidth || v.size() != rhs.v.size())
      return false;

    for (unsigned long i = 0; i < v.size(); ++i)
      if (v[i] != rhs.v[i])
        return false;

    return true;
  };

  bool isTop() const { return *this == top(bitwidth); }
  bool isBottom() const { return *this == bottom(bitwidth); }
  bool isSuperset(const D &rhs) const { return meet(rhs) == rhs; }
  unsigned int distance(const D &rhs) const {
    return (v[0] ^ rhs.v[0]).popcount() + (v[1] ^ rhs.v[1]).popcount();
  }

  static const D toBestAbst(const D &lhs, const D &rhs,
                            uint8_t (*op)(const uint8_t, const uint8_t),
                            uint8_t bw) {
    uint64_t mask = makeMask(bw);
    std::vector<D> crtVals;

    for (auto lhs_v : lhs.toConcrete()) {
      for (auto rhs_v : rhs.toConcrete()) {
        // stubbed out op_constraint for now
        // if (op_constraint(APInt(bitwidth, lhs_v), APInt(bitwidth, rhs_v))) {}
        llvm::APInt v(bw, op(lhs_v, rhs_v) & mask);
        crtVals.push_back(AbstVal<D>::fromConcrete(v));
      }
    }

    return AbstVal<D>::joinAll(bw, crtVals);
  }

  virtual ~AbstVal() = default;
  virtual bool isConstant() const = 0;
  virtual llvm::APInt getConstant() const = 0;
  virtual D meet(const D &) const = 0;
  virtual D join(const D &) const = 0;
  virtual std::vector<unsigned char> const toConcrete() const = 0;
  virtual std::string display() const = 0;
};

// class KnownBits : public KnownBits<AbstVal<N>, N>
class KnownBits : public AbstVal<KnownBits> {
private:
  llvm::APInt zero() const { return v[0]; }
  llvm::APInt one() const { return v[1]; }
  bool hasConflict() const { return zero().intersects(one()); }

public:
  // TODO would prefer if this was private
  KnownBits(const std::vector<llvm::APInt> &v, unsigned int bw)
      : AbstVal(v, bw) {
    // TODO auto return bottom whenever an invalid KB is trying to be
    // constructed (if one intersects zero)
  }

  virtual std::string display() const override {
    if (isBottom()) {
      return "(bottom)";
    }

    std::stringstream ss;

    for (unsigned int i = bitwidth; i > 0; --i) {
      ss << (one()[i - 1] ? '1' : zero()[i - 1] ? '0' : '?');
    }

    if (isConstant())
      ss << getConstant().getZExtValue();

    if (isTop())
      ss << " (top)";

    return ss.str();
  }

  virtual bool isConstant() const override { return zero() == one(); }
  virtual llvm::APInt getConstant() const override { return zero(); }

  virtual KnownBits meet(const KnownBits &rhs) const override {
    return KnownBits({zero() | rhs.zero(), one() | rhs.one()}, bitwidth);
  }

  virtual KnownBits join(const KnownBits &rhs) const override {
    return KnownBits({zero() & rhs.zero(), one() & rhs.one()}, bitwidth);
  }

  // TODO there should be a much faster way to do this
  virtual std::vector<unsigned char> const toConcrete() const override {
    std::vector<unsigned char> ret;
    const llvm::APInt min = llvm::APInt::getZero(bitwidth);
    const llvm::APInt max = llvm::APInt::getMaxValue(bitwidth);

    for (auto i = min;; ++i) {

      if (!zero().intersects(i) && !one().intersects(~i))
        ret.push_back(static_cast<unsigned char>(i.getZExtValue()));

      if (i == max)
        break;
    }

    return ret;
  }

  static KnownBits fromConcrete(const llvm::APInt &x) {
    return KnownBits({~x, x}, x.getBitWidth());
  }

  static KnownBits bottom(unsigned int bw) {
    llvm::APInt max = llvm::APInt::getMaxValue(bw);
    return KnownBits({max, max}, 0);
  }

  static KnownBits top(unsigned int bw) {
    llvm::APInt min = llvm::APInt::getMinValue(bw);
    return KnownBits({min, min}, 0);
  }

  // TODO there should be a faster way to do this
  static std::vector<KnownBits> const enumVals(const unsigned int bw) {
    std::vector<KnownBits> ret;
    const llvm::APInt max = llvm::APInt::getMaxValue(bw);
    for (unsigned long i = 0; i <= max.getZExtValue(); ++i) {
      for (unsigned long j = 0; j <= max.getZExtValue(); ++j) {
        llvm::APInt zero = llvm::APInt(bw, i);
        llvm::APInt one = llvm::APInt(bw, j);
        KnownBits x({zero, one}, bw);

        if (!x.hasConflict())
          ret.push_back(x);
      }
    }
    return ret;
  }
};
