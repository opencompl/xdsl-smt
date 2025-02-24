#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
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
        AbstVal x(KNOWN_BITS, {zero, one}, bitwidth);

        if (!x.hasConflict())
          ret.push_back(x);
      }
    }
    return ret;
  } else if (d == CONSTANT_RANGE) {
    // TODO there should be some speed wins here
    const llvm::APInt min = llvm::APInt::getMinValue(bitwidth);
    const llvm::APInt max = llvm::APInt::getMaxValue(bitwidth);
    std::vector<AbstVal> ret = {AbstVal::top(CONSTANT_RANGE, bitwidth)};

    for (llvm::APInt i = min;; ++i) {
      for (llvm::APInt j = min;; ++j) {
        if (j.ult(i))
          continue;

        ret.push_back(AbstVal(CONSTANT_RANGE, {i, j}, bitwidth));

        if (j == max)
          break;
      }
      if (i == max)
        break;
    }

    return ret;
  } else{
    assert(false);
  }

  fprintf(stderr, "unknown abstract domain\n");
}

uint64_t makeMask(uint8_t bitwidth) {
  if (bitwidth == 0)
    return 0;
  return (1 << bitwidth) - 1;
}

AbstVal toBestAbstract(const AbstVal lhs, const AbstVal rhs,
                       uint8_t (*op)(const uint8_t, const uint8_t),
                       uint8_t bitwidth, Domain d) {
  assert(lhs.domain == rhs.domain && "lhs and rhs must be in the same domain");
  assert((lhs.domain == KNOWN_BITS || lhs.domain == CONSTANT_RANGE) &&
         "function not implemented for other domains");

  uint64_t mask = makeMask(bitwidth);
  std::vector<AbstVal> crtVals;

  for (auto lhs_v : lhs.toConcrete()) {
    for (auto rhs_v : rhs.toConcrete()) {
      // stubbed out op_constraint for now
      // if (op_constraint(APInt(bitwidth, lhs_v), APInt(bitwidth, rhs_v))) {}
      llvm::APInt v(bitwidth, op(lhs_v, rhs_v) & mask);
      crtVals.push_back(AbstVal::fromConcrete(d, v));
    }
  }

  return AbstVal::joinAll(d, bitwidth, crtVals);
}

int main(int argv, char **argc) {
  // TODO make a flag for bitwidth
  const size_t bitwidth = 4;
  if (argv != 3 || strcmp(argc[1], "--domain") != 0) {
    fprintf(stderr, "usage: ./EvalEngine --domain KnownBits\n");
    return 1;
  }

  Domain d;
  if (strcmp(argc[2], "KnownBits") == 0) {
    d = KNOWN_BITS;
  } else if (strcmp(argc[2], "ConstantRange") == 0) {
    d = CONSTANT_RANGE;
  } else {
    fprintf(stderr, "Error unknown domain: %s\n", argc[2]);
    return 1;
  }

  // TODO maybe make this a cmd line flag but idk
  Results r{numFuncs};

  for (auto lhs : enumAbstVals(bitwidth, d)) {
    for (auto rhs : enumAbstVals(bitwidth, d)) {

      auto best_abstract_res =
          toBestAbstract(lhs, rhs, concrete_op_wrapper, bitwidth, d);

      std::vector<AbstVal> synth_kbs(synth_function_wrapper(lhs, rhs));
      std::vector<AbstVal> ref_kbs(ref_function_wrapper(lhs, rhs));
      // join of all kb values in the vec, ref_kbs
      AbstVal cur_kb = AbstVal::meetAll(d, bitwidth, ref_kbs);
      bool solved = cur_kb == best_abstract_res;

      for (unsigned int i = 0; i < synth_kbs.size(); ++i) {
        AbstVal synth_after_meet = cur_kb.meet(synth_kbs[i]);
        bool sound = synth_after_meet.isSuperset(best_abstract_res);
        bool exact = synth_after_meet == best_abstract_res;
        // TODO distance is kind of a bogus measure of CONST_RANGE
        unsigned int dis = synth_after_meet.distance(best_abstract_res);

        r.incResult(Result(sound, dis, exact, solved), i);
      }

      r.incCases(solved);
    }
  }

  r.print();

  return 0;
}
