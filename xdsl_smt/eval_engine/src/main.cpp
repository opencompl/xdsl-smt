#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <llvm/ADT/APInt.h>

#include "AbstVal.cpp"
#include "Results.cpp"
#include "synth.cpp"

// TODO there's a faster way to this but this works for now
// would also be nice if this moved up the lattice as the loops progressed
// TODO x2 need to exclude certain abstract vals as needed
// TODO x3 probs should be put into `AbstVal.cpp`
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
  // TODO fix this crap
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

// check if res is a superset of best_res
// TODO probs put in `AbstVal.cpp`
bool kb_check_include(const AbstVal &res, const AbstVal &best_res) {
  return res.unionWith(best_res) == best_res;
}

// compute the edit distance between 2 KnownBits
// TODO probs put in `AbstVal.cpp`
unsigned int kb_edit_dis(const AbstVal &res, const AbstVal &best_res) {
  return (res.v[0] ^ best_res.v[0]).popcount() +
         (res.v[1] ^ best_res.v[1]).popcount();
}

int main() {
  const size_t bitwidth = 4;
  Results r{numFuncs};

  for (auto lhs : enumAbstVals(bitwidth, KNOWN_BITS)) {
    for (auto rhs : enumAbstVals(bitwidth, KNOWN_BITS)) {

      auto best_abstract_res =
          toBestAbstract(lhs, rhs, concrete_op_wrapper, bitwidth);

      std::vector<AbstVal> synth_kbs(synth_function_wrapper(lhs, rhs));
      std::vector<AbstVal> ref_kbs(ref_function_wrapper(lhs, rhs));

      // this creates a bottom val for kb
      AbstVal cur_kb(KNOWN_BITS, bitwidth);
      // then cur_kb is unioned with all elems in ref_kbs
      // TODO put a function in `AbstVal.cpp` to do this from a vec of abstvals
      for (auto kb : ref_kbs)
        cur_kb = cur_kb.unionWith(kb);

      bool solved = cur_kb == best_abstract_res;

      for (unsigned int i = 0; i < synth_kbs.size(); ++i) {
        AbstVal synth_after_meet = cur_kb.unionWith(synth_kbs[i]);
        bool sound = kb_check_include(synth_after_meet, best_abstract_res);
        bool exact = synth_after_meet == best_abstract_res;
        unsigned int dis = kb_edit_dis(synth_after_meet, best_abstract_res);

        r.incResult(Result(sound, dis, exact, solved), i);
      }

      r.incCases(solved);
    }
  }

  r.print();

  return 0;
}
