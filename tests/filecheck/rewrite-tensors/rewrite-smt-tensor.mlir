// RUN: xdsl-smt "%s" -p=rewrite-smt-tensor | filecheck "%s"

// Rewrite tensor from  "smt.extract" operations.

builtin.module {
  %0 = "smt.declare_const"() : () -> !smt.tensor.tensor<[3, 3], !smt.fp<8, 24>, none>
  %idx1 = "smt.declare_const"() : () -> !smt.bv<64>
  %idx2 = "smt.declare_const"() : () -> !smt.bv<64>
  %transpose = "smt.tensor.transpose"(%0){permutation = [1, 0]} :
        (!smt.tensor.tensor<[3, 3], !smt.fp<8, 24>, none>) -> !smt.tensor.tensor<[3, 3], !smt.fp<8, 24>, none>
  %extract = "smt.tensor.extract"(%transpose, %idx1, %idx2):
        (!smt.tensor.tensor<[3, 3], !smt.fp<8, 24>, none>, !smt.bv<64>, !smt.bv<64>) -> !smt.fp<8, 24>
}
