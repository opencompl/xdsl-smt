#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <llvm/ADT/APInt.h>

#include "AbstVal.cpp"
#include "Results.cpp"
#include "synth.cpp"

// TODO there's a faster way to this but this works for now
// would also be nice if this moved up the lattice as the loops progressed
// TODO x2 need to exclude certain abstract vals as needed
//
// TODO x3 probs should be put into `AbstVal.cpp`
// std::vector<AbstVal> const enumAbstVals(const uint32_t bitwidth,
//                                         const Domain d) {
//   } else if (d == CONSTANT_RANGE) {
//     // TODO there should be some speed wins here
//     const llvm::APInt min = llvm::APInt::getMinValue(bitwidth);
//     const llvm::APInt max = llvm::APInt::getMaxValue(bitwidth);
//     std::vector<AbstVal> ret = {AbstVal::top(CONSTANT_RANGE, bitwidth)};
//
//     for (llvm::APInt i = min;; ++i) {
//       for (llvm::APInt j = min;; ++j) {
//         if (j.ult(i))
//           continue;
//
//         ret.push_back(AbstVal(CONSTANT_RANGE, {i, j}, bitwidth));
//
//         if (j == max)
//           break;
//       }
//       if (i == max)
//         break;
//     }
//
//     return ret;
//   }
// }

template <typename Domain> Results eval(unsigned int bw, unsigned int nFuncs) {
  Results r{nFuncs};

  for (auto lhs : Domain::enumVals(bw)) {
    for (auto rhs : Domain::enumVals(bw)) {

      auto best_abstract_res = Domain::toBestAbst(lhs, rhs, concrete_op_wrapper,
                                                  static_cast<uint8_t>(bw));

      std::vector<Domain> synth_kbs(synth_function_wrapper(lhs, rhs));
      std::vector<Domain> ref_kbs(ref_function_wrapper(lhs, rhs));
      Domain cur_kb = Domain::meetAll(bw, ref_kbs);
      bool solved = cur_kb == best_abstract_res;

      for (unsigned int i = 0; i < synth_kbs.size(); ++i) {
        Domain synth_after_meet = cur_kb.meet(synth_kbs[i]);
        bool sound = synth_after_meet.isSuperset(best_abstract_res);
        bool exact = synth_after_meet == best_abstract_res;
        unsigned int dis = synth_after_meet.distance(best_abstract_res);

        r.incResult(Result(sound, dis, exact, solved), i);
      }

      r.incCases(solved);
    }
  }

  return r;
}

int main(int argv, char **argc) {
  // TODO maybe make bitwidth a cmd line flag but idk

  // TODO make a flag for bitwidth
  const size_t bitwidth = 4;
  if (argv != 3 || strcmp(argc[1], "--domain") != 0) {
    fprintf(stderr, "usage: ./EvalEngine --domain KnownBits\n");
    return 1;
  }

  // TODO there is probs a good way to select the domain at compile time
  if (strcmp(argc[2], "KnownBits") == 0) {
    Results r = eval<KnownBits>(bitwidth, numFuncs);
    r.print();
  } else if (strcmp(argc[2], "ConstantRange") == 0) {
    printf("whoops");
  } else {
    fprintf(stderr, "Error unknown domain: %s\n", argc[2]);
    return 1;
  }

  return 0;
}
