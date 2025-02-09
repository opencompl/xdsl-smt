#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
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
    printf(" const %llu", x.getConstant().getZExtValue());

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

llvm::KnownBits to_best_abstract(const llvm::KnownBits lhs,
                                 const llvm::KnownBits rhs,
                                 uint8_t (*op)(const uint8_t, const uint8_t),
                                 uint8_t bitwidth) {
  bool hasInit = false;
  llvm::KnownBits result(bitwidth);
  uint8_t mask = 0b00001111;
  for (auto lhs_val : to_concrete(lhs)) {
    for (auto rhs_val : to_concrete(rhs)) {
      if (op_constraint(APInt(bitwidth, lhs_val), APInt(bitwidth, rhs_val))) {
        auto crt_res = llvm::KnownBits::makeConstant(
            llvm::APInt(bitwidth, op(lhs_val, rhs_val) & mask));
        if (!hasInit) {
          result = crt_res;
          hasInit = true;
        } else {
          result = result.intersectWith(crt_res);
        }
      }
    }
  }
  return result;
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

unsigned int compare_abstract(llvm::KnownBits abs_res,
                              llvm::KnownBits best_abs_res, bool &isUnsound) {
  const llvm::APInt min = llvm::APInt::getZero(abs_res.Zero.getBitWidth());
  const llvm::APInt max = llvm::APInt::getMaxValue(abs_res.Zero.getBitWidth());

  unsigned result = 0;

  for (auto i = min;; ++i) {
    bool in_abs_res =
        !abs_res.Zero.intersects(i) && !abs_res.One.intersects(~i);
    bool in_best_abs_res =
        !best_abs_res.Zero.intersects(i) && !best_abs_res.One.intersects(~i);
    if (in_best_abs_res && !in_abs_res) {
      // unsound
      isUnsound = true;
    } else if (in_abs_res && !in_best_abs_res) {
      ++result;
    }
    if (i == max)
      break;
  }

  return result;
}

// check if res is a superset of best_res
bool kb_check_include(const llvm::KnownBits &res,
                      const llvm::KnownBits &best_res) {
  return res.unionWith(best_res) == best_res;
}

// compute the edit distance between 2 KnownBits
int kb_edit_dis(const llvm::KnownBits &res, const llvm::KnownBits &best_res) {
  return (res.Zero ^ best_res.Zero).popcount() +
         (res.One ^ best_res.One).popcount();
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

  std::vector<std::vector<unsigned int>> all_cases;
  std::vector<std::vector<unsigned int>> unsolved_cases;

  long long total_abst_combos = 0;
  long long total_unsolved_combos = 0;

  for (auto lhs : enum_abst_vals(bitwidth)) {
    for (auto rhs : enum_abst_vals(bitwidth)) {
      auto best_abstract_res =
          to_best_abstract(lhs, rhs, concrete_op_wrapper, bitwidth);

      std::vector<llvm::KnownBits> synth_kbs(synth_function_wrapper(lhs, rhs));
      std::vector<llvm::KnownBits> ref_kbs(ref_function_wrapper(lhs, rhs));

      llvm::KnownBits cur_kb = llvm::KnownBits(bitwidth);
      for (auto kb : ref_kbs)
        cur_kb = cur_kb.unionWith(kb);

      if (all_cases.size() == 0) {
        all_cases.resize(synth_kbs.size());
        std::fill(all_cases.begin(), all_cases.end(),
                  std::vector<unsigned int>{0, 0, 0, 0});
      }

      if (unsolved_cases.size() == 0) {
        unsolved_cases.resize(synth_kbs.size());
        std::fill(unsolved_cases.begin(), unsolved_cases.end(),
                  std::vector<unsigned int>{0, 0, 0, 0});
      }

      bool solved = cur_kb == best_abstract_res;

      for (int i = 0; i < synth_kbs.size(); ++i) {
        llvm::KnownBits synth_after_meet = cur_kb.unionWith(synth_kbs[i]);
        bool sound = kb_check_include(synth_after_meet, best_abstract_res);
        bool exact = synth_after_meet == best_abstract_res;
        int dis = kb_edit_dis(synth_after_meet, best_abstract_res);

        all_cases[i][0] += sound;
        all_cases[i][1] += dis;
        all_cases[i][2] += exact;
        if (!solved) {
          unsolved_cases[i][0] += sound;
          unsolved_cases[i][1] += dis;
          unsolved_cases[i][2] += exact;
        }

        // std::clog << "lhs: ";
        // lhs.print(llvm::dbgs());
        // std::clog << ", rhs: ";
        // rhs.print(llvm::dbgs());
        // std::clog << ", best: ";
        // best_abstract_res.print(llvm::dbgs());
        // std::clog << ", res: ";
        // synth_after_meet.print(llvm::dbgs());
        // std::clog << ", sound: " << sound << ", exact: " << exact
        //           << ", dis: " << dis << "\n";
      }

      total_abst_combos++;
      if (!solved)
        total_unsolved_combos++;
    }
  }

  for (auto &res : all_cases) {
    res[3] = total_abst_combos;
  }

  for (auto &res : unsolved_cases) {
    res[3] = total_unsolved_combos;
  }

  // printf("Not sound or precise: %i\n", cases[0]);
  // printf("Not sound:            %i\n", cases[1]);
  // printf("Not precise:          %i\n", cases[2]);
  // printf("Good:                 %i\n", cases[3]);
  // printf("total tests: %lld\n", total_abst_combos);

  printf("sound:\n[");
  for (int i = 0; i < all_cases.size(); ++i)
    printf("%d, ", all_cases[i][0]);
  printf("]\n");

  printf("precise:\n[");
  for (int i = 0; i < all_cases.size(); ++i)
    printf("%d, ", all_cases[i][1]);
  printf("]\n");

  printf("exact:\n[");
  for (int i = 0; i < all_cases.size(); ++i)
    printf("%d, ", all_cases[i][2]);
  printf("]\n");

  printf("num_cases:\n[");
  for (int i = 0; i < all_cases.size(); ++i)
    printf("%d, ", all_cases[i][3]);
  printf("]\n");

  printf("unsolved_sound:\n[");
  for (int i = 0; i < unsolved_cases.size(); ++i)
    printf("%d, ", unsolved_cases[i][0]);
  printf("]\n");

  printf("unsolved_precise:\n[");
  for (int i = 0; i < unsolved_cases.size(); ++i)
    printf("%d, ", unsolved_cases[i][1]);
  printf("]\n");

  printf("unsolved_exact:\n[");
  for (int i = 0; i < unsolved_cases.size(); ++i)
    printf("%d, ", unsolved_cases[i][2]);
  printf("]\n");

  printf("unsolved_num_cases:\n[");
  for (int i = 0; i < unsolved_cases.size(); ++i)
    printf("%d, ", unsolved_cases[i][3]);
  printf("]\n");

  return 0;
}
