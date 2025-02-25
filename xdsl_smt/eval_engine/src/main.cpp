#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <llvm/ADT/APInt.h>

#include "Results.cpp"
#include "synth.cpp"

unsigned int makeMask(unsigned char bitwidth) {
  if (bitwidth == 0)
    return 0;
  return (1 << bitwidth) - 1;
}

template <typename Domain>
const Domain toBestAbst(const Domain &lhs, const Domain &rhs,
                        unsigned int (*op)(const unsigned int,
                                           const unsigned int)) {
  const unsigned char bitwidth = lhs.getBitWidth();
  llvm::APInt x(bitwidth, 0);
  llvm::APInt lhs_op_con(bitwidth, 0);
  llvm::APInt rhs_op_con(bitwidth, 0);
  unsigned int mask = makeMask(bitwidth);
  std::vector<Domain> crtVals;
  const std::vector<unsigned int> rhss = rhs.toConcrete();

  for (unsigned int lhs_v : lhs.toConcrete()) {
    for (unsigned int rhs_v : rhss) {
      lhs_op_con = lhs_v;
      rhs_op_con = rhs_v;
      if (op_constraint(lhs_op_con, rhs_op_con)) {
        x = op(lhs_v, rhs_v) & mask;
        crtVals.push_back(Domain::fromConcrete(x));
      }
    }
  }

  return Domain::joinAll(crtVals);
}

template <typename Domain> const Results eval(unsigned int nFuncs) {
  Results r{nFuncs};
  const std::vector<Domain> fullLattice = Domain::enumVals();

  for (Domain lhs : fullLattice) {
    for (Domain rhs : fullLattice) {
      Domain best_abstract_res = toBestAbst(lhs, rhs, concrete_op_wrapper);
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
