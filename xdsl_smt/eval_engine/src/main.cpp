#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <llvm/ADT/APInt.h>

#include "Results.cpp"
#include "synth.cpp"

template <typename Domain> Results eval(unsigned int nFuncs) {
  Results r{nFuncs};
  const std::vector<Domain> fullLattice = Domain::enumVals();

  for (Domain lhs : fullLattice) {
    for (Domain rhs : fullLattice) {
      Domain best_abstract_res = lhs.toBestAbst(rhs, concrete_op_wrapper);
      std::vector<Domain> synth_kbs(synth_function_wrapper(lhs, rhs));
      std::vector<Domain> ref_kbs(ref_function_wrapper(lhs, rhs));
      Domain cur_kb = Domain::meetAll(ref_kbs);
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

int main() {
  eval<Domain>(numFuncs).print();
  return 0;
}
