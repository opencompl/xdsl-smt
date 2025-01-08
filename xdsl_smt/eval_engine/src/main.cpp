#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <llvm/ADT/APInt.h>
#include <llvm/Support/KnownBits.h>
#include <vector>

#include "synth.cpp"

// TODO switch between signed and unsigned ops when needed
// enum TransferResult { GOOD, NOT_SOUND, NOT_PREC, NEITHER };

void print_abst_range(const llvm::KnownBits &x) {
  for (uint32_t i = x.Zero.getBitWidth() - 1; i >= 0; --i) {
    const char bit = x.One[i] ? '1' : x.Zero[i] ? '0' : '?';
    printf("%c", bit);
  }

  if (x.isConstant())
    printf(" const %lu", x.getConstant().getZExtValue());

  if (x.isUnknown())
    printf(" (top)");

  printf("\n");
}

// TODO consider printing full/top if it is
void print_conc_range(const std::vector<uint8_t> &x) {
  if (x.empty())
    printf("empty");

  for (auto i : x)
    printf("%hhu ", i);

  puts("");
}

// TODO there's a faster way to this but this works for now
// would also be nice if this moved up the lattice as the loops progressed
std::vector<llvm::KnownBits> const enum_abst_vals(const uint32_t bitwidth) {
  std::vector<llvm::KnownBits> ret;
  const llvm::APInt max = llvm::APInt::getMaxValue(bitwidth);
  for (uint64_t i = 0; i <= max.getZExtValue(); ++i) {
    for (uint64_t j = 0; j <= max.getZExtValue(); ++j) {
      auto x = llvm::KnownBits(bitwidth);
      x.One = i;
      x.Zero = j;

      if (!x.hasConflict())
        ret.push_back(x);
    }
  }

  return ret;
}

// TODO return a generic container based on what the caller asks for
// TODO there's a faster way to this but this works for now
std::vector<uint8_t> const to_concrete(const llvm::KnownBits &x) {
  std::vector<uint8_t> ret;
  const llvm::APInt min = llvm::APInt::getZero(x.Zero.getBitWidth());
  const llvm::APInt max = llvm::APInt::getMaxValue(x.Zero.getBitWidth());

  for (auto i = min;; ++i) {

    if (!x.Zero.intersects(i) && !x.One.intersects(~i))
      ret.push_back((uint8_t)i.getZExtValue());

    if (i == max)
      break;
  }

  return ret;
}

llvm::KnownBits const to_abstract(const std::vector<uint8_t> &conc_vals,
                                  uint8_t bitwidth) {
  auto ret = llvm::KnownBits::makeConstant(llvm::APInt(bitwidth, conc_vals[0]));

  for (auto x : conc_vals) {
    ret = ret.intersectWith(
        llvm::KnownBits::makeConstant(llvm::APInt(bitwidth, x)));
  }

  return ret;
}

// TODO be able to return generic std container
// TODO have some automated check for UB?
// TODO auto vary bitmask based on width and spec consideration for signed ints
std::vector<uint8_t> const concrete_op_enum(const std::vector<uint8_t> &lhss,
                                            const std::vector<uint8_t> &rhss,
                                            uint8_t (*op)(const uint8_t,
                                                          const uint8_t)) {
  auto ret = std::vector<uint8_t>();

  uint8_t mask = 0b00001111;

  for (auto lhs : lhss)
    for (auto rhs : rhss)
      ret.push_back(op(lhs, rhs) & mask);

  // remove duplicates
  std::sort(ret.begin(), ret.end());
  ret.erase(unique(ret.begin(), ret.end()), ret.end());

  return ret;
}

// TODO make case enum
unsigned int compare(std::vector<uint8_t> &approx,
                     std::vector<uint8_t> &exact) {

  bool sound = true;
  bool prec = true;
  std::sort(approx.begin(), approx.end());
  std::sort(exact.begin(), exact.end());

  std::vector<uint64_t> approx_m_exact;
  std::set_difference(approx.begin(), approx.end(), exact.begin(), exact.end(),
                      std::back_inserter(approx_m_exact));

  if (!approx_m_exact.empty())
    prec = false;

  std::vector<uint64_t> exact_m_approx;
  std::set_difference(exact.begin(), exact.end(), approx.begin(), approx.end(),
                      std::back_inserter(exact_m_approx));

  if (!exact_m_approx.empty())
    sound = false;

  if (!sound && !prec)
    return 0;

  if (!sound)
    return 1;

  if (!prec)
    return 2;

  return 3;
}

int main() {
  const size_t bitwidth = 4;

  std::vector<int> cases = {0, 0, 0, 0};
  long long total_cases = 0;

  for (auto lhs : enum_abst_vals(bitwidth)) {
    for (auto rhs : enum_abst_vals(bitwidth)) {
      auto transfer_vals = to_concrete(synth_function_wrapper(lhs, rhs));
      auto brute_vals =
          concrete_op_enum(to_concrete(lhs), to_concrete(rhs), concrete_op);

      const unsigned int caseNum = compare(transfer_vals, brute_vals);
      cases[caseNum]++;
      total_cases++;
    }
  }

  double percent_sound = (double)(cases[3] + cases[2]) / (double)total_cases;
  double percent_precise = (double)(cases[3] + cases[1]) / (double)total_cases;

  printf("Not sound or precise: %i\n", cases[0]);
  printf("Not sound:            %i\n", cases[1]);
  printf("Not precise:          %i\n", cases[2]);
  printf("Good:                 %i\n", cases[3]);
  printf("total tests:          %lld\n", total_cases);
  printf("sound percent:        %f\n", percent_sound);
  printf("precise percent:      %f\n", percent_precise);

  return 0;
}
