#ifndef APInt_H
#define APInt_H

namespace A {
class APInt;
inline APInt operator-(APInt);

class [[nodiscard]] APInt {
private:
  static unsigned rotateModulo(unsigned BitWidth, const APInt &rotateAmt) {
    if (BitWidth == 0)
      return 0;
    unsigned rotBitWidth = rotateAmt.getBitWidth();
    APInt rot = rotateAmt;
    if (rotBitWidth < BitWidth) {
      rot = rotateAmt.zext(BitWidth);
    }
    rot = rot.urem(APInt(rot.getBitWidth(), BitWidth));
    return static_cast<unsigned int>(rot.getLimitedValue(BitWidth));
  }

  static inline long SignExtend64(unsigned long X, unsigned B) {
    if (B == 0)
      return 0;
    return long(X << (64 - B)) >> (64 - B);
  }

public:
  typedef unsigned long WordType;
  static constexpr unsigned long WORDTYPE_MAX = ~static_cast<unsigned long>(0);
  static constexpr unsigned APINT_WORD_SIZE = sizeof(unsigned long);
  static constexpr unsigned APINT_BITS_PER_WORD = APINT_WORD_SIZE * 8;

  explicit APInt() {
    VAL = 0;
    BitWidth = 1;
  }

  APInt(unsigned numBits, unsigned long val) : BitWidth(numBits), VAL(val) {
    clearUnusedBits();
  }

  /// Copy Constructor.
  APInt(const APInt &that) : BitWidth(that.BitWidth) { VAL = that.VAL; }

  static APInt getZero(unsigned numBits) { return APInt(numBits, 0); }
  static APInt getZeroWidth() { return getZero(0); }
  static APInt getMaxValue(unsigned numBits) { return getAllOnes(numBits); }
  static APInt getMinValue(unsigned numBits) { return APInt(numBits, 0); }

  static APInt getSignedMaxValue(unsigned numBits) {
    APInt API = getAllOnes(numBits);
    API.clearBit(numBits - 1);
    return API;
  }

  static APInt getSignedMinValue(unsigned numBits) {
    APInt API(numBits, 0);
    API.setBit(numBits - 1);
    return API;
  }

  static APInt getSignMask(unsigned BitWidth) {
    return getSignedMinValue(BitWidth);
  }

  static APInt getAllOnes(unsigned numBits) {
    return APInt(numBits, WORDTYPE_MAX);
  }

  static APInt getOneBitSet(unsigned numBits, unsigned BitNo) {
    APInt Res(numBits, 0);
    Res.setBit(BitNo);
    return Res;
  }

  static APInt getSplat(unsigned NewLen, const APInt &V) {
    APInt Val = V.zext(NewLen);
    for (unsigned I = V.getBitWidth(); I < NewLen; I <<= 1)
      Val |= Val << I;

    return Val;
  }

  bool isNegative() const { return (*this)[BitWidth - 1]; }
  bool isNonNegative() const { return !isNegative(); }
  bool isSignBitSet() const { return (*this)[BitWidth - 1]; }
  bool isSignBitClear() const { return !isSignBitSet(); }
  bool isStrictlyPositive() const { return isNonNegative() && !isZero(); }
  bool isNonPositive() const { return !isStrictlyPositive(); }
  bool isZero() const { return VAL == 0; }
  bool isOne() const { return VAL == 1; }
  bool isMaxValue() const { return isAllOnes(); }
  bool isMinValue() const { return isZero(); }
  bool isIntN(unsigned N) const { return getActiveBits() <= N; }
  bool isSignedIntN(unsigned N) const { return getSignificantBits() <= N; }
  bool isSignMask() const { return isMinSignedValue(); }
  bool getBoolValue() const { return !isZero(); }
  bool isSplat(unsigned SplatSizeInBits) const {
    return *this == rotl(SplatSizeInBits);
  }

  APInt getHiBits(unsigned numBits) const {
    return this->lshr(BitWidth - numBits);
  }

  APInt getLoBits(unsigned numBits) const;
  bool isOneBitSet(unsigned BitNo) const {
    return (*this)[BitNo] && popcount() == 1;
  }

  bool isAllOnes() const {
    return VAL == WORDTYPE_MAX >> (APINT_BITS_PER_WORD - BitWidth);
  }

  bool isMaxSignedValue() const {
    return VAL == ((WordType(1) << (BitWidth - 1)) - 1);
  }

  bool isMinSignedValue() const {
    return VAL == (WordType(1) << (BitWidth - 1));
  }

  bool isNegatedPowerOf2() const {
    if (isNonNegative())
      return false;
    // NegatedPowerOf2 - shifted mask in the top bits.
    unsigned LO = countl_one();
    unsigned TZ = countr_zero();
    return (LO + TZ) == BitWidth;
  }

  unsigned long getLimitedValue(unsigned long Limit = WORDTYPE_MAX) const {
    return ugt(Limit) ? Limit : getZExtValue();
  }

  bool isMask(unsigned numBits) const {
    return VAL == (WORDTYPE_MAX >> (APINT_BITS_PER_WORD - numBits));
  }

  static bool isSameValue(const APInt &I1, const APInt &I2) {
    if (I1.getBitWidth() == I2.getBitWidth())
      return I1 == I2;

    if (I1.getBitWidth() > I2.getBitWidth())
      return I1 == I2.zext(I1.getBitWidth());

    return I1.zext(I2.getBitWidth()) == I2;
  }

  APInt operator++(int) {
    APInt API(*this);
    ++(*this);
    return API;
  }

  APInt &operator++() {
    ++VAL;
    return clearUnusedBits();
  }

  APInt operator--(int) {
    APInt API(*this);
    --(*this);
    return API;
  }

  APInt &operator--() {
    --VAL;
    return clearUnusedBits();
  }

  bool operator!() const { return isZero(); }

  APInt &operator=(const APInt &RHS) {
    VAL = RHS.VAL;
    BitWidth = RHS.BitWidth;
    return *this;
  }

  APInt &operator=(APInt &&that) {
    VAL = that.BitWidth;
    BitWidth = that.BitWidth;
    that.BitWidth = 0;
    return *this;
  }

  APInt &operator=(unsigned long RHS) {
    VAL = RHS;
    return clearUnusedBits();
  }

  APInt &operator&=(const APInt &RHS) {
    VAL &= RHS.VAL;
    return *this;
  }

  APInt &operator&=(unsigned long RHS) {
    VAL &= RHS;
    return *this;
  }

  APInt &operator|=(const APInt &RHS) {
    VAL |= RHS.VAL;
    return *this;
  }

  APInt &operator|=(unsigned long RHS) {
    VAL |= RHS;
    return clearUnusedBits();
  }

  APInt &operator^=(const APInt &RHS) {
    VAL ^= RHS.VAL;
    return *this;
  }

  APInt &operator^=(unsigned long RHS) {
    VAL ^= RHS;
    return clearUnusedBits();
  }

  APInt &operator*=(const APInt &RHS) {
    *this = *this * RHS;
    return *this;
  }

  APInt &operator*=(unsigned long RHS) {
    VAL *= RHS;
    return clearUnusedBits();
  }

  APInt &operator+=(const APInt &RHS) {
    VAL += RHS.VAL;
    return clearUnusedBits();
  }

  APInt &operator+=(unsigned long RHS) {
    VAL += RHS;
    return clearUnusedBits();
  }

  APInt &operator-=(const APInt &RHS) {
    VAL -= RHS.VAL;
    return clearUnusedBits();
  }

  APInt &operator-=(unsigned long RHS) {
    VAL -= RHS;
    return clearUnusedBits();
  }

  APInt &operator<<=(unsigned ShiftAmt) {
    if (ShiftAmt == BitWidth)
      VAL = 0;
    else
      VAL <<= ShiftAmt;
    return clearUnusedBits();
    return *this;
  }

  APInt &operator<<=(const APInt &shiftAmt) {
    // It's undefined behavior in C to shift by BitWidth or greater.
    *this <<= static_cast<unsigned>(shiftAmt.getLimitedValue(BitWidth));
    return *this;
  }

  APInt operator*(const APInt &RHS) const {
    return APInt(BitWidth, VAL * RHS.VAL);
  }

  APInt operator<<(unsigned Bits) const { return shl(Bits); }
  APInt operator<<(const APInt &Bits) const { return shl(Bits); }

  APInt ashr(unsigned ShiftAmt) const {
    APInt R(*this);
    R.ashrInPlace(ShiftAmt);
    return R;
  }

  void ashrInPlace(unsigned ShiftAmt) {
    long SExtVAL = SignExtend64(VAL, BitWidth);
    if (ShiftAmt == BitWidth)
      VAL = static_cast<unsigned long>(SExtVAL >> (APINT_BITS_PER_WORD - 1));
    else
      VAL = static_cast<unsigned long>(SExtVAL >> ShiftAmt);
    clearUnusedBits();
    return;
  }

  APInt lshr(unsigned shiftAmt) const {
    APInt R(*this);
    R.lshrInPlace(shiftAmt);
    return R;
  }

  void lshrInPlace(unsigned ShiftAmt) {
    if (ShiftAmt == BitWidth)
      VAL = 0;
    else
      VAL >>= ShiftAmt;
    return;
  }

  APInt shl(unsigned shiftAmt) const {
    APInt R(*this);
    R <<= shiftAmt;
    return R;
  }

  APInt relativeLShr(int RelativeShift) const {
    return RelativeShift > 0 ? lshr(static_cast<unsigned int>(RelativeShift))
                             : shl(static_cast<unsigned int>(-RelativeShift));
  }

  APInt relativeLShl(int RelativeShift) const {
    return relativeLShr(-RelativeShift);
  }

  APInt relativeAShr(int RelativeShift) const {
    return RelativeShift > 0 ? ashr(static_cast<unsigned int>(RelativeShift))
                             : shl(static_cast<unsigned int>(-RelativeShift));
  }

  APInt relativeAShl(int RelativeShift) const {
    return relativeAShr(-RelativeShift);
  }

  APInt rotl(unsigned rotateAmt) const;
  APInt rotr(unsigned rotateAmt) const;

  APInt ashr(const APInt &ShiftAmt) const {
    APInt R(*this);
    R.ashrInPlace(ShiftAmt);
    return R;
  }

  void ashrInPlace(const APInt &shiftAmt) {
    ashrInPlace(static_cast<unsigned>(shiftAmt.getLimitedValue(BitWidth)));
  }

  APInt lshr(const APInt &ShiftAmt) const {
    APInt R(*this);
    R.lshrInPlace(ShiftAmt);
    return R;
  }

  void lshrInPlace(const APInt &shiftAmt) {
    lshrInPlace(static_cast<unsigned>(shiftAmt.getLimitedValue(BitWidth)));
  }

  APInt shl(const APInt &ShiftAmt) const {
    APInt R(*this);
    R <<= ShiftAmt;
    return R;
  }

  APInt rotl(const APInt &rotateAmt) const {
    return rotl(rotateModulo(BitWidth, rotateAmt));
  }
  APInt rotr(const APInt &rotateAmt) const {
    return rotr(rotateModulo(BitWidth, rotateAmt));
  }
  APInt udiv(const APInt &RHS) const { return APInt(BitWidth, VAL / RHS.VAL); }

  APInt udiv(unsigned long RHS) const { return APInt(BitWidth, VAL / RHS); }

  APInt sdiv(const APInt &RHS) const {
    if (isNegative()) {
      if (RHS.isNegative())
        return (-(*this)).udiv(-RHS);
      return -((-(*this)).udiv(RHS));
    }
    if (RHS.isNegative())
      return -(this->udiv(-RHS));
    return this->udiv(RHS);
  }

  APInt sdiv(long RHS) const {
    if (isNegative()) {
      if (RHS < 0)
        return (-(*this)).udiv(static_cast<unsigned long>(-RHS));
      return -((-(*this)).udiv(static_cast<unsigned long>(RHS)));
    }
    if (RHS < 0)
      return -(this->udiv(static_cast<unsigned long>(-RHS)));
    return this->udiv(static_cast<unsigned long>(RHS));
  }

  APInt urem(const APInt &RHS) const { return APInt(BitWidth, VAL % RHS.VAL); }

  unsigned long urem(unsigned long RHS) const { return VAL % RHS; }

  APInt srem(const APInt &RHS) const {
    if (isNegative()) {
      if (RHS.isNegative())
        return -((-(*this)).urem(-RHS));
      return -((-(*this)).urem(RHS));
    }
    if (RHS.isNegative())
      return this->urem(-RHS);
    return this->urem(RHS);
  }

  long srem(long RHS) const {
    if (isNegative()) {
      if (RHS < 0)
        return static_cast<long>(
            -((-(*this)).urem(static_cast<unsigned long>(-RHS))));
      return static_cast<long>(
          -((-(*this)).urem(static_cast<unsigned long>(RHS))));
    }
    if (RHS < 0)
      return static_cast<long>(this->urem(static_cast<unsigned long>(-RHS)));
    return static_cast<long>(this->urem(static_cast<unsigned long>(RHS)));
  }

  static void udivrem(const APInt &LHS, const APInt &RHS, APInt &Quotient,
                      APInt &Remainder) {
    unsigned BitWidth = LHS.BitWidth;
    unsigned long QuotVal = LHS.VAL / RHS.VAL;
    unsigned long RemVal = LHS.VAL % RHS.VAL;
    Quotient = APInt(BitWidth, QuotVal);
    Remainder = APInt(BitWidth, RemVal);
    return;
  }

  static void udivrem(const APInt &LHS, unsigned long RHS, APInt &Quotient,
                      unsigned long &Remainder) {
    unsigned BitWidth = LHS.BitWidth;
    unsigned long QuotVal = LHS.VAL / RHS;
    Remainder = LHS.VAL % RHS;
    Quotient = APInt(BitWidth, QuotVal);
    return;
  }

  static void sdivrem(const APInt &LHS, const APInt &RHS, APInt &Quotient,
                      APInt &Remainder) {
    if (LHS.isNegative()) {
      if (RHS.isNegative())
        APInt::udivrem(-LHS, -RHS, Quotient, Remainder);
      else {
        APInt::udivrem(-LHS, RHS, Quotient, Remainder);
        Quotient.negate();
      }
      Remainder.negate();
    } else if (RHS.isNegative()) {
      APInt::udivrem(LHS, -RHS, Quotient, Remainder);
      Quotient.negate();
    } else {
      APInt::udivrem(LHS, RHS, Quotient, Remainder);
    }
  }

  static void sdivrem(const APInt &LHS, long RHS, APInt &Quotient,
                      long &Remainder) {
    unsigned long R = static_cast<unsigned long>(Remainder);
    if (LHS.isNegative()) {
      if (RHS < 0)
        APInt::udivrem(-LHS, static_cast<unsigned long>(-RHS), Quotient, R);
      else {
        APInt::udivrem(-LHS, static_cast<unsigned long>(RHS), Quotient, R);
        Quotient.negate();
      }
      R = -R;
    } else if (RHS < 0) {
      APInt::udivrem(LHS, static_cast<unsigned long>(-RHS), Quotient, R);
      Quotient.negate();
    } else {
      APInt::udivrem(LHS, static_cast<unsigned long>(RHS), Quotient, R);
    }
    Remainder = static_cast<long>(R);
  }

  APInt sadd_ov(const APInt &RHS, bool &Overflow) const;
  APInt uadd_ov(const APInt &RHS, bool &Overflow) const;
  APInt ssub_ov(const APInt &RHS, bool &Overflow) const;
  APInt usub_ov(const APInt &RHS, bool &Overflow) const;
  APInt sdiv_ov(const APInt &RHS, bool &Overflow) const {
    // MININT/-1  -->  overflow.
    Overflow = isMinSignedValue() && RHS.isAllOnes();
    return sdiv(RHS);
  }

  APInt smul_ov(const APInt &RHS, bool &Overflow) const {
    APInt Res = *this * RHS;

    if (RHS != 0)
      Overflow =
          Res.sdiv(RHS) != *this || (isMinSignedValue() && RHS.isAllOnes());
    else
      Overflow = false;
    return Res;
  }

  APInt umul_ov(const APInt &RHS, bool &Overflow) const {
    if (countl_zero() + RHS.countl_zero() + 2 <= BitWidth) {
      Overflow = true;
      return *this * RHS;
    }

    APInt Res = lshr(1) * RHS;
    Overflow = Res.isNegative();
    Res <<= 1;
    if ((*this)[0]) {
      Res += RHS;
      if (Res.ult(RHS))
        Overflow = true;
    }
    return Res;
  }

  APInt sshl_ov(const APInt &ShAmt, bool &Overflow) const {
    return sshl_ov(
        static_cast<unsigned int>(ShAmt.getLimitedValue(getBitWidth())),
        Overflow);
  }

  APInt sshl_ov(unsigned ShAmt, bool &Overflow) const {
    Overflow = ShAmt >= getBitWidth();
    if (Overflow)
      return APInt(BitWidth, 0);

    if (isNonNegative()) // Don't allow sign change.
      Overflow = ShAmt >= countl_zero();
    else
      Overflow = ShAmt >= countl_one();

    return *this << ShAmt;
  }

  APInt ushl_ov(const APInt &ShAmt, bool &Overflow) const {
    return ushl_ov(
        static_cast<unsigned int>(ShAmt.getLimitedValue(getBitWidth())),
        Overflow);
  }

  APInt ushl_ov(unsigned ShAmt, bool &Overflow) const {
    Overflow = ShAmt >= getBitWidth();
    if (Overflow)
      return APInt(BitWidth, 0);

    Overflow = ShAmt > countl_zero();

    return *this << ShAmt;
  }

  APInt sfloordiv_ov(const APInt &RHS, bool &Overflow) const;
  APInt sadd_sat(const APInt &RHS) const {
    bool Overflow;
    APInt Res = sadd_ov(RHS, Overflow);
    if (!Overflow)
      return Res;

    return isNegative() ? APInt::getSignedMinValue(BitWidth)
                        : APInt::getSignedMaxValue(BitWidth);
  }

  APInt uadd_sat(const APInt &RHS) const {
    bool Overflow;
    APInt Res = uadd_ov(RHS, Overflow);
    if (!Overflow)
      return Res;

    return APInt::getMaxValue(BitWidth);
  }

  APInt ssub_sat(const APInt &RHS) const {
    bool Overflow;
    APInt Res = ssub_ov(RHS, Overflow);
    if (!Overflow)
      return Res;

    return isNegative() ? APInt::getSignedMinValue(BitWidth)
                        : APInt::getSignedMaxValue(BitWidth);
  }

  APInt usub_sat(const APInt &RHS) const {
    bool Overflow;
    APInt Res = usub_ov(RHS, Overflow);
    if (!Overflow)
      return Res;

    return APInt(BitWidth, 0);
  }

  APInt smul_sat(const APInt &RHS) const {
    bool Overflow;
    APInt Res = smul_ov(RHS, Overflow);
    if (!Overflow)
      return Res;

    // The result is negative if one and only one of inputs is negative.
    bool ResIsNegative = isNegative() ^ RHS.isNegative();

    return ResIsNegative ? APInt::getSignedMinValue(BitWidth)
                         : APInt::getSignedMaxValue(BitWidth);
  }

  APInt umul_sat(const APInt &RHS) const {
    bool Overflow;
    APInt Res = umul_ov(RHS, Overflow);
    if (!Overflow)
      return Res;

    return APInt::getMaxValue(BitWidth);
  }

  APInt sshl_sat(const APInt &RHS) const {
    return sshl_sat(
        static_cast<unsigned int>(RHS.getLimitedValue(getBitWidth())));
  }

  APInt sshl_sat(unsigned RHS) const {
    bool Overflow;
    APInt Res = sshl_ov(RHS, Overflow);
    if (!Overflow)
      return Res;

    return isNegative() ? APInt::getSignedMinValue(BitWidth)
                        : APInt::getSignedMaxValue(BitWidth);
  }

  APInt ushl_sat(const APInt &RHS) const {
    return ushl_sat(
        static_cast<unsigned int>(RHS.getLimitedValue(getBitWidth())));
  }

  APInt ushl_sat(unsigned RHS) const {
    bool Overflow;
    APInt Res = ushl_ov(RHS, Overflow);
    if (!Overflow)
      return Res;

    return APInt::getMaxValue(BitWidth);
  }

  bool operator[](unsigned bitPosition) const {
    return (maskBit(bitPosition) & VAL) != 0;
  }

  bool operator==(const APInt &RHS) const { return VAL == RHS.VAL; }
  bool operator==(unsigned long Val) const { return getZExtValue() == Val; }
  bool eq(const APInt &RHS) const { return (*this) == RHS; }
  bool operator!=(const APInt &RHS) const { return !((*this) == RHS); }
  bool operator!=(unsigned long Val) const { return !((*this) == Val); }
  bool ne(const APInt &RHS) const { return !((*this) == RHS); }
  bool ult(const APInt &RHS) const { return compare(RHS) < 0; }
  bool ult(unsigned long RHS) const { return getZExtValue() < RHS; }
  bool slt(const APInt &RHS) const { return compareSigned(RHS) < 0; }
  bool slt(long RHS) const { return getSExtValue() < RHS; }
  bool ule(const APInt &RHS) const { return compare(RHS) <= 0; }
  bool ule(unsigned long RHS) const { return !ugt(RHS); }
  bool sle(const APInt &RHS) const { return compareSigned(RHS) <= 0; }
  bool sle(unsigned long RHS) const { return !sgt(static_cast<long>(RHS)); }
  bool ugt(const APInt &RHS) const { return !ule(RHS); }
  bool ugt(unsigned long RHS) const { return getZExtValue() > RHS; }
  bool sgt(const APInt &RHS) const { return !sle(RHS); }
  bool sgt(long RHS) const { return getSExtValue() > RHS; }
  bool uge(const APInt &RHS) const { return !ult(RHS); }
  bool uge(unsigned long RHS) const { return !ult(RHS); }
  bool sge(const APInt &RHS) const { return !slt(RHS); }
  bool sge(long RHS) const { return !slt(RHS); }
  bool intersects(const APInt &RHS) const { return (VAL & RHS.VAL) != 0; }
  bool isSubsetOf(const APInt &RHS) const { return (VAL & ~RHS.VAL) == 0; }

  APInt trunc(unsigned width) const { return APInt(width, VAL); }

  APInt truncUSat(unsigned width) const {
    // Can we just losslessly truncate it?
    if (isIntN(width))
      return trunc(width);
    // If not, then just return the new limit.
    return APInt::getMaxValue(width);
  }

  APInt truncSSat(unsigned width) const {
    // Can we just losslessly truncate it?
    if (isSignedIntN(width))
      return trunc(width);
    // If not, then just return the new limits.
    return isNegative() ? APInt::getSignedMinValue(width)
                        : APInt::getSignedMaxValue(width);
  }

  APInt zext(unsigned width) const { return APInt(width, VAL); }

  APInt sext(unsigned Width) const {
    return APInt(Width,
                 static_cast<unsigned long>(SignExtend64(VAL, BitWidth)));
  }

  APInt zextOrTrunc(unsigned width) const {
    if (BitWidth < width)
      return zext(width);
    if (BitWidth > width)
      return trunc(width);
    return *this;
  }

  APInt sextOrTrunc(unsigned width) const {
    if (BitWidth < width)
      return sext(width);
    if (BitWidth > width)
      return trunc(width);
    return *this;
  }

  void setAllBits() {
    VAL = WORDTYPE_MAX;
    clearUnusedBits();
  }

  void setBit(unsigned BitPosition) {
    WordType Mask = maskBit(BitPosition);
    VAL |= Mask;
  }

  void setSignBit() { setBit(BitWidth - 1); }

  void setBitVal(unsigned BitPosition, bool BitValue) {
    if (BitValue)
      setBit(BitPosition);
    else
      clearBit(BitPosition);
  }

  void clearAllBits() { VAL = 0; }

  void clearBit(unsigned BitPosition) {
    WordType Mask = ~maskBit(BitPosition);
    VAL &= Mask;
  }

  void clearSignBit() { clearBit(BitWidth - 1); }

  void flipAllBits() {
    VAL ^= WORDTYPE_MAX;
    clearUnusedBits();
  }

  void negate() {
    flipAllBits();
    ++(*this);
  }

  void flipBit(unsigned bitPosition) {
    setBitVal(bitPosition, !(*this)[bitPosition]);
  }

  void insertBits(const APInt &SubBits, unsigned bitPosition);
  void insertBits(unsigned long SubBits, unsigned bitPosition,
                  unsigned numBits);
  APInt extractBits(unsigned numBits, unsigned bitPosition) const;
  unsigned long extractBitsAsZExtValue(unsigned numBits,
                                       unsigned bitPosition) const;
  unsigned getBitWidth() const { return BitWidth; }
  unsigned getNumWords() const { return 1; }
  unsigned getActiveBits() const { return BitWidth - countl_zero(); }

  unsigned getSignificantBits() const {
    return BitWidth - getNumSignBits() + 1;
  }

  unsigned long getZExtValue() const { return VAL; }
  long getSExtValue() const { return SignExtend64(VAL, BitWidth); }

  unsigned countl_zero() const {
    unsigned unusedBits = APINT_BITS_PER_WORD - BitWidth;
    return (VAL == 0 ? 64 : static_cast<unsigned int>(__builtin_clzll(VAL))) -
           unusedBits;
  }

  unsigned countl_one() const {
    unsigned long tmp = ~(VAL << (APINT_BITS_PER_WORD - BitWidth));
    return tmp == 0 ? 64 : static_cast<unsigned int>(__builtin_clzll(tmp));
  }

  unsigned getNumSignBits() const {
    return isNegative() ? countl_one() : countl_zero();
  }

  unsigned countr_zero() const {
    unsigned TrailingZeros = static_cast<unsigned int>(__builtin_ctzll(VAL));
    return (TrailingZeros > BitWidth ? BitWidth : TrailingZeros);
  }

  unsigned countLeadingZeros() const { return countl_zero(); }
  unsigned countLeadingOnes() const { return countl_one(); }
  unsigned countTrailingZeros() const { return countr_zero(); }
  unsigned countr_one() const {
    return static_cast<unsigned int>(__builtin_ctzll(~VAL));
  }
  unsigned countTrailingOnes() const { return countr_one(); }
  unsigned popcount() const {
    return static_cast<unsigned int>(__builtin_popcountll(VAL));
  }
  APInt byteSwap() const;
  APInt reverseBits() const;
  double roundToDouble(bool isSigned) const;
  double roundToDouble() const { return roundToDouble(false); }
  double signedRoundToDouble() const { return roundToDouble(true); }
  unsigned logBase2() const { return getActiveBits() - 1; }

  unsigned ceilLogBase2() const {
    APInt temp(*this);
    --temp;
    return temp.getActiveBits();
  }

  APInt abs() const {
    if (isNegative())
      return -(*this);
    return *this;
  }

  APInt multiplicativeInverse() const;

private:
  unsigned BitWidth;
  unsigned long VAL;

  static unsigned long maskBit(unsigned bitPosition) {
    return 1ULL << bitPosition;
  }

  APInt &clearUnusedBits() {
    // Compute how many bits are used in the final word.
    unsigned WordBits = ((BitWidth - 1) % APINT_BITS_PER_WORD) + 1;

    // Mask out the high bits.
    unsigned long mask = WORDTYPE_MAX >> (APINT_BITS_PER_WORD - WordBits);
    if (BitWidth == 0)
      mask = 0;

    VAL &= mask;
    return *this;
  }

  int compare(const APInt &RHS) const {
    return VAL < RHS.VAL ? -1 : VAL > RHS.VAL;
  }

  int compareSigned(const APInt &RHS) const {
    long lhsSext = SignExtend64(VAL, BitWidth);
    long rhsSext = SignExtend64(RHS.VAL, BitWidth);
    return lhsSext < rhsSext ? -1 : lhsSext > rhsSext;
  }
};

inline bool operator==(unsigned long V1, const APInt &V2) { return V2 == V1; }
inline bool operator!=(unsigned long V1, const APInt &V2) { return V2 != V1; }
inline APInt operator~(APInt v) {
  v.flipAllBits();
  return v;
}

inline APInt operator&(APInt a, const APInt &b) {
  a &= b;
  return a;
}

inline APInt operator&(const APInt &a, APInt &&b) {
  b &= a;
  return b;
}

inline APInt operator&(APInt a, unsigned long RHS) {
  a &= RHS;
  return a;
}

inline APInt operator&(unsigned long LHS, APInt b) {
  b &= LHS;
  return b;
}

inline APInt operator|(APInt a, const APInt &b) {
  a |= b;
  return a;
}

inline APInt operator|(const APInt &a, APInt &&b) {
  b |= a;
  return b;
}

inline APInt operator|(APInt a, unsigned long RHS) {
  a |= RHS;
  return a;
}

inline APInt operator|(unsigned long LHS, APInt b) {
  b |= LHS;
  return b;
}

inline APInt operator^(APInt a, const APInt &b) {
  a ^= b;
  return a;
}

inline APInt operator^(const APInt &a, APInt &&b) {
  b ^= a;
  return b;
}

inline APInt operator^(APInt a, unsigned long RHS) {
  a ^= RHS;
  return a;
}

inline APInt operator^(unsigned long LHS, APInt b) {
  b ^= LHS;
  return b;
}
inline APInt operator+(APInt a, const APInt &b) {
  a += b;
  return a;
}

inline APInt operator+(const APInt &a, APInt &&b) {
  b += a;
  return b;
}

inline APInt operator+(APInt a, unsigned long RHS) {
  a += RHS;
  return a;
}

inline APInt operator+(unsigned long LHS, APInt b) {
  b += LHS;
  return b;
}

inline APInt operator-(APInt a, const APInt &b) {
  a -= b;
  return a;
}

inline APInt operator-(const APInt &a, APInt &&b) {
  b.negate();
  b += a;
  return b;
}

inline APInt operator-(APInt a, unsigned long RHS) {
  a -= RHS;
  return a;
}

inline APInt operator-(unsigned long LHS, APInt b) {
  b.negate();
  b += LHS;
  return b;
}

inline APInt operator*(APInt a, unsigned long RHS) {
  a *= RHS;
  return a;
}

inline APInt operator*(unsigned long LHS, APInt b) {
  b *= LHS;
  return b;
}

inline APInt operator-(APInt v) {
  v.negate();
  return v;
}

namespace APIntOps {
/// Determine the smaller of two APInts considered to be signed.
inline const APInt &smin(const APInt &A, const APInt &B) {
  return A.slt(B) ? A : B;
}

/// Determine the larger of two APInts considered to be signed.
inline const APInt &smax(const APInt &A, const APInt &B) {
  return A.sgt(B) ? A : B;
}

/// Determine the smaller of two APInts considered to be unsigned.
inline const APInt &umin(const APInt &A, const APInt &B) {
  return A.ult(B) ? A : B;
}

/// Determine the larger of two APInts considered to be unsigned.
inline const APInt &umax(const APInt &A, const APInt &B) {
  return A.ugt(B) ? A : B;
}

/// Determine the absolute difference of two APInts considered to be signed.
inline const APInt abds(const APInt &A, const APInt &B) {
  return A.sge(B) ? (A - B) : (B - A);
}

/// Determine the absolute difference of two APInts considered to be unsigned.
inline const APInt abdu(const APInt &A, const APInt &B) {
  return A.uge(B) ? (A - B) : (B - A);
}

inline APInt avgFloorS(const APInt &C1, const APInt &C2) {
  // Return floor((C1 + C2) / 2)
  return (C1 & C2) + (C1 ^ C2).ashr(1);
}

inline APInt avgFloorU(const APInt &C1, const APInt &C2) {
  // Return floor((C1 + C2) / 2)
  return (C1 & C2) + (C1 ^ C2).lshr(1);
}

inline APInt avgCeilS(const APInt &C1, const APInt &C2) {
  // Return ceil((C1 + C2) / 2)
  return (C1 | C2) - (C1 ^ C2).ashr(1);
}

inline APInt avgCeilU(const APInt &C1, const APInt &C2) {
  // Return ceil((C1 + C2) / 2)
  return (C1 | C2) - (C1 ^ C2).lshr(1);
}

} // namespace APIntOps

///////////////////////////////////////////////////
/// APInt.cpp

inline APInt APInt::rotl(unsigned rotateAmt) const {
  if (BitWidth == 0)
    return *this;
  rotateAmt %= BitWidth;
  if (rotateAmt == 0)
    return *this;
  return shl(rotateAmt) | lshr(BitWidth - rotateAmt);
}

inline APInt APInt::rotr(unsigned rotateAmt) const {
  if (BitWidth == 0)
    return *this;
  rotateAmt %= BitWidth;
  if (rotateAmt == 0)
    return *this;
  return lshr(rotateAmt) | shl(BitWidth - rotateAmt);
}

/// returns the multiplicative inverse of an odd APInt modulo 2^BitWidth.
inline APInt APInt::multiplicativeInverse() const {
  // Use Newton's method.
  APInt Factor = *this;
  APInt T;
  while (!(T = *this * Factor).isOne())
    Factor *= 2 - T;
  return Factor;
}

inline APInt APInt::sadd_ov(const APInt &RHS, bool &Overflow) const {
  APInt Res = *this + RHS;
  Overflow = isNonNegative() == RHS.isNonNegative() &&
             Res.isNonNegative() != isNonNegative();
  return Res;
}

inline APInt APInt::uadd_ov(const APInt &RHS, bool &Overflow) const {
  APInt Res = *this + RHS;
  Overflow = Res.ult(RHS);
  return Res;
}

inline APInt APInt::ssub_ov(const APInt &RHS, bool &Overflow) const {
  APInt Res = *this - RHS;
  Overflow = isNonNegative() != RHS.isNonNegative() &&
             Res.isNonNegative() != isNonNegative();
  return Res;
}

inline APInt APInt::usub_ov(const APInt &RHS, bool &Overflow) const {
  APInt Res = *this - RHS;
  Overflow = Res.ugt(*this);
  return Res;
}

inline APInt APInt::sfloordiv_ov(const APInt &RHS, bool &Overflow) const {
  APInt quotient = sdiv_ov(RHS, Overflow);
  if ((quotient * RHS != *this) && (isNegative() != RHS.isNegative()))
    return quotient - 1;
  return quotient;
}

} // namespace A

template <unsigned int N> class Vec {
public:
  A::APInt v[N];
  unsigned int getN() const { return N; }

  template <typename... Args> Vec(Args... args) {
    static_assert(sizeof...(args) == N, "Number of arguments must match N");
    A::APInt arr[] = {args...};

    for (unsigned int i = 0; i < N; ++i)
      v[i] = arr[i];
  }

  Vec(const A::APInt *x) {
    for (unsigned int i = 0; i < N; ++i)
      v[i] = x[i];
  }

  const A::APInt &operator[](unsigned int i) const { return v[i]; }
};

#endif
