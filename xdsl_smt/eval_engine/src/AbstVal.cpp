#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

#include <llvm/ADT/APInt.h>

consteval unsigned int makeMask(unsigned char bitwidth) {
  if (bitwidth == 0)
    return 0;
  return (1 << bitwidth) - 1;
}

template <typename Domain, unsigned char N> class AbstVal {
protected:
  explicit AbstVal(const std::vector<llvm::APInt> &v) : v(v) {}
public:
  std::vector<llvm::APInt> v;

public:
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

  static const Domain fromConcrete(const llvm::APInt &x) {
    static_assert(std::is_base_of<AbstVal, Domain>::value);
    static_assert(std::is_same<decltype(&Domain::fromConcrete),
                               Domain (*)(const llvm::APInt &)>::value);
    return Domain::fromConcrete(x);
  }

  static const Domain bottom() {
    static_assert(std::is_base_of<AbstVal, Domain>::value);
    static_assert(std::is_same<decltype(&Domain::bottom), Domain (*)()>::value);
    return Domain::bottom();
  }

  static const Domain top() {
    static_assert(std::is_base_of<AbstVal, Domain>::value);
    static_assert(std::is_same<decltype(&Domain::top), Domain (*)()>::value);

    return Domain::top();
  }

  static const std::vector<Domain> enumVals() {
    static_assert(std::is_base_of<AbstVal, Domain>::value);
    static_assert(
        std::is_same<decltype(&Domain::enumVals), Domain (*)()>::value);
    return Domain::enumVals();
  }

  bool operator==(const AbstVal &rhs) const {
    if (v.size() != rhs.v.size())
      return false;

    for (unsigned long i = 0; i < v.size(); ++i)
      if (v[i] != rhs.v[i])
        return false;

    return true;
  };

  bool isTop() const { return *this == top(); }
  bool isBottom() const { return *this == bottom(); }
  bool isSuperset(const Domain &rhs) const { return meet(rhs) == rhs; }
  unsigned int distance(const Domain &rhs) const {
    return (v[0] ^ rhs.v[0]).popcount() + (v[1] ^ rhs.v[1]).popcount();
  }

  static const Domain toBestAbst(const Domain &lhs, const Domain &rhs,
                                 unsigned int (*op)(const unsigned int,
                                                    const unsigned int)) {
    unsigned int mask = makeMask(N);
    std::vector<Domain> crtVals;

    for (auto lhs_v : lhs.toConcrete()) {
      for (auto rhs_v : rhs.toConcrete()) {
        llvm::APInt v(N, op(lhs_v, rhs_v) & mask);
        crtVals.push_back(AbstVal<Domain, N>::fromConcrete(v));
      }
    }

    return AbstVal<Domain, N>::joinAll(crtVals);
  }

  virtual ~AbstVal() = default;
  virtual bool isConstant() const = 0;
  virtual llvm::APInt getConstant() const = 0;
  virtual Domain meet(const Domain &) const = 0;
  virtual Domain join(const Domain &) const = 0;
  virtual std::vector<unsigned int> const toConcrete() const = 0;
  virtual std::string display() const = 0;
};

template <unsigned char N> class KnownBits : public AbstVal<KnownBits<N>, N> {
private:
  llvm::APInt zero() const { return this->v[0]; }
  llvm::APInt one() const { return this->v[1]; }
  bool hasConflict() const { return zero().intersects(one()); }

public:
  explicit KnownBits(const std::vector<llvm::APInt> &v)
      : AbstVal<KnownBits<N>, N>(v) {}

  virtual std::string display() const override {
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

  virtual bool isConstant() const override { return zero() == one(); }
  virtual llvm::APInt getConstant() const override { return zero(); }

  virtual KnownBits meet(const KnownBits &rhs) const override {
    return KnownBits({zero() | rhs.zero(), one() | rhs.one()});
  }

  virtual KnownBits join(const KnownBits &rhs) const override {
    return KnownBits({zero() & rhs.zero(), one() & rhs.one()});
  }

  // TODO there should be a much faster way to do this
  virtual std::vector<unsigned int> const toConcrete() const override {
    std::vector<unsigned int> ret;
    const llvm::APInt min = llvm::APInt::getZero(N);
    const llvm::APInt max = llvm::APInt::getMaxValue(N);

    for (auto i = min;; ++i) {

      if (!zero().intersects(i) && !one().intersects(~i))
        ret.push_back(i.getZExtValue());

      if (i == max)
        break;
    }

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

  // TODO there should be a faster way to do this
  static std::vector<KnownBits> const enumVals() {
    std::vector<KnownBits> ret;
    const llvm::APInt max = llvm::APInt::getMaxValue(N);
    for (unsigned long i = 0; i <= max.getZExtValue(); ++i) {
      for (unsigned long j = 0; j <= max.getZExtValue(); ++j) {
        llvm::APInt zero = llvm::APInt(N, i);
        llvm::APInt one = llvm::APInt(N, j);
        KnownBits x({zero, one});

        if (!x.hasConflict())
          ret.push_back(x);
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

  virtual std::string display() const override {
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

  virtual bool isConstant() const override { return lower() == upper(); }
  virtual llvm::APInt getConstant() const override { return lower(); }

  virtual ConstantRange meet(const ConstantRange &rhs) const override {
    llvm::APInt l = rhs.lower().ugt(lower()) ? rhs.lower() : lower();
    llvm::APInt u = rhs.upper().ult(upper()) ? rhs.upper() : upper();
    if (l.ugt(u))
      return bottom();
    return ConstantRange({std::move(l), std::move(u)});
  }

  virtual ConstantRange join(const ConstantRange &rhs) const override {
    llvm::APInt l = rhs.lower().ult(lower()) ? rhs.lower() : lower();
    llvm::APInt u = rhs.upper().ugt(upper()) ? rhs.upper() : upper();
    return ConstantRange({std::move(l), std::move(u)});
  }

  virtual std::vector<unsigned int> const toConcrete() const override {
    std::vector<unsigned int> ret;
    unsigned int l = lower().getZExtValue();
    unsigned int u = upper().getZExtValue() + 1;

    if (l > u)
      return {};

    return std::views::iota(l, u) | std::ranges::to<std::vector>();
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

  // TODO there should be a faster way to do this
  static std::vector<ConstantRange> const enumVals() {
    const llvm::APInt min = llvm::APInt::getMinValue(N);
    const llvm::APInt max = llvm::APInt::getMaxValue(N);
    std::vector<ConstantRange> ret = {top()};

    for (llvm::APInt i = min;; ++i) {
      for (llvm::APInt j = min;; ++j) {
        if (j.ult(i))
          continue;

        ret.push_back(ConstantRange({i, j}));

        if (j == max)
          break;
      }
      if (i == max)
        break;
    }

    return ret;
  }
};
