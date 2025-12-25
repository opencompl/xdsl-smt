// RUN: xdsl-smt "%s" -p=lower-smt-tensor -t=smt | filecheck "%s"

// Lower tensor from  "smt.tensor" operations.

builtin.module {
  %0 = "smt.declare_const"() : () -> !smt.tensor.tensor<[3, 3], !smt.fp<8, 24>, none>
  // CHECK:      (declare-const $tmp (Array (_ BitVec 64) (Array (_ BitVec 64) (_ FloatingPoint 8 24))

  %idx1 = "smt.declare_const"() : () -> !smt.bv<64>
  %idx2 = "smt.declare_const"() : () -> !smt.bv<64>
  // CHECK-NEXT: (declare-const $idx1 (_ BitVec 64))
  // CHECK-NEXT: (declare-const $idx2 (_ BitVec 64))

  %extract = "smt.tensor.extract"(%0, %idx1, %idx2):
        (!smt.tensor.tensor<[3, 3], !smt.fp<8, 24>, none>, !smt.bv<64>, !smt.bv<64>) -> !smt.fp<8, 24>
  %zero = "smt.fp.pzero"() : () -> !smt.fp<8,24>
  %eq_zero = "smt.eq"(%extract, %zero) : (!smt.fp<8,24>, !smt.fp<8,24>) -> !smt.bool
  "smt.assert"(%eq_zero) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= (select (select $tmp $idx1) $idx2) (_ +zero 8 24)))
}
