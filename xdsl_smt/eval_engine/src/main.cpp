#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <llvm/ADT/APInt.h>

#include "AbstVal.cpp"
#include "synth.cpp"

// TODO consider printing full/top if it is
void printConcRange(const std::vector<uint8_t> &x) {
  if (x.empty())
    printf("empty");

  for (auto i : x)
    printf("%hhu ", i);

  printf("\n");
}

// TODO there's a faster way to this but this works for now
// would also be nice if this moved up the lattice as the loops progressed
// TODO x2 need to exclude certain abstract vals as needed
std::vector<AbstVal> const enumAbstVals(const uint32_t bitwidth,
                                        const Domain d) {
  if (d == KNOWN_BITS) {
    std::vector<AbstVal> ret;
    const llvm::APInt max = llvm::APInt::getMaxValue(bitwidth);
    for (uint64_t i = 0; i <= max.getZExtValue(); ++i) {
      for (uint64_t j = 0; j <= max.getZExtValue(); ++j) {
        llvm::APInt zero = llvm::APInt(bitwidth, i);
        llvm::APInt one = llvm::APInt(bitwidth, j);
        AbstVal x(KNOWN_BITS, {zero, one});

        if (!x.hasConflict())
          ret.push_back(x);
      }
    }
    return ret;
  } else if (d == CONSTANT_RANGE) {
    std::vector<AbstVal> ret;
    const llvm::APInt min = llvm::APInt::getMinValue(bitwidth);
    const llvm::APInt max = llvm::APInt::getMaxValue(bitwidth);

    for (llvm::APInt i = min;; ++i) {
      for (llvm::APInt j = min;; ++j) {
        if (i == j && !(i.isMaxValue() || i.isMinValue())) {
          if (j == max)
            break;
          else
            continue;
        }

        ret.push_back(AbstVal(CONSTANT_RANGE, {i, j}));

        if (j == max)
          break;
      }
      if (i == max)
        break;
    }

    return ret;
  } else {
    printf("unknown abstract domain\n");
  }

  return {};
}

AbstVal toBestAbstract(const AbstVal lhs, const AbstVal rhs,
                       uint8_t (*op)(const uint8_t, const uint8_t),
                       uint8_t bitwidth) {

  assert(lhs.domain == KNOWN_BITS && rhs.domain == KNOWN_BITS &&
         "function not implemented for other domains\n");

  // TODO generate this bit mask automaticlly
  uint8_t mask = 0b00001111;
  // really incredibly stupid but idk how to use unique ptrs
  std::vector<AbstVal> res;

  for (auto lhs_val : lhs.toConcrete()) {
    for (auto rhs_val : rhs.toConcrete()) {
      // stubbed out op_constraint for now
      // if (op_constraint(APInt(bitwidth, lhs_val), APInt(bitwidth, rhs_val)))
      if (true) {
        llvm::APInt v(bitwidth, op(lhs_val, rhs_val) & mask);
        AbstVal crtVal(KNOWN_BITS, v);

        if (res.size() == 0) {
          res.push_back(crtVal);
        } else {
          res[0] = res[0].intersectWith(crtVal);
        }
      }
    }
  }

  return res[0];
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

unsigned int compare_abstract(AbstVal abs_res, AbstVal best_abs_res,
                              bool &isUnsound, uint32_t bitwidth) {
  assert(abs_res.domain == KNOWN_BITS && best_abs_res.domain == KNOWN_BITS &&
         "function not implemented for other domains\n");

  const llvm::APInt min = llvm::APInt::getMinValue(bitwidth);
  const llvm::APInt max = llvm::APInt::getMaxValue(bitwidth);

  unsigned result = 0;

  for (auto i = min;; ++i) {
    bool in_abs_res =
        !abs_res.v[0].intersects(i) && !abs_res.v[1].intersects(~i);
    bool in_best_abs_res =
        !best_abs_res.v[0].intersects(i) && !best_abs_res.v[1].intersects(~i);
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

void printEvalResults(const std::vector<std::vector<unsigned int>> &x) {
  printf("sound:\n");
  printf("[");
  for (uint32_t i = 0; i < x.size(); ++i) {
    printf("%d, ", x[i][0]);
  }
  printf("]\n");

  printf("precise:\n");
  printf("[");
  for (uint32_t i = 0; i < x.size(); ++i) {
    printf("%d, ", x[i][1]);
  }
  printf("]\n");

  printf("exact:\n");
  printf("[");
  for (uint32_t i = 0; i < x.size(); ++i) {
    printf("%d, ", x[i][2]);
  }
  printf("]\n");

  printf("num_cases:\n");
  printf("[");
  for (uint32_t i = 0; i < x.size(); ++i) {
    printf("%d, ", x[i][3]);
  }
  printf("]\n");
}

int main() {
  const size_t bitwidth = 4;

  std::vector<std::vector<unsigned int>> all_cases;
  long long total_abst_combos = 0;

  for (auto lhs : enumAbstVals(bitwidth, KNOWN_BITS)) {
    for (auto rhs : enumAbstVals(bitwidth, KNOWN_BITS)) {

      auto best_abstract_res =
          toBestAbstract(lhs, rhs, concrete_op_wrapper, bitwidth);

      std::vector<AbstVal> synth_kbs(synth_function_wrapper(lhs, rhs));

      if (all_cases.size() == 0) {
        all_cases.resize(synth_kbs.size());
        std::fill(all_cases.begin(), all_cases.end(),
                  std::vector<unsigned int>{0, 0, 0, 0});
      }

      for (uint32_t i = 0; i < synth_kbs.size(); ++i) {
        // sound non_precision exact num_cases
        bool isUnsound = false;
        if (synth_kbs[i] == best_abstract_res) {
          all_cases[i][2] += 1;
        } else {
          auto num_non_precision = compare_abstract(
              synth_kbs[i], best_abstract_res, isUnsound, bitwidth);
          if (isUnsound) {
            all_cases[i][0] += 1;
          }
          all_cases[i][1] += num_non_precision;
        }
      }

      total_abst_combos++;
    }
  }
  for (auto &res : all_cases) {
    res[3] = static_cast<uint32_t>(total_abst_combos);
  }

  printEvalResults(all_cases);

  return 0;
}
