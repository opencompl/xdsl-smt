// RUN: xdsl-smt.py %s | xdsl-smt.py -t=smt | filecheck %s
// RUN: xdsl-smt.py %s -t=smt | z3 -in


builtin.module {
  %x = "smt.declare_const"() : () -> !smt.bv.bv<5>
  %y = "smt.declare_const"() : () -> !smt.bv.bv<10>
  %z = "smt.declare_const"() : () -> !smt.bv.bv<15>
  // CHECK:      (declare-const x (_ BitVec 5))
  // CHECK-NEXT: (declare-const y (_ BitVec 10))
  // CHECK-NEXT: (declare-const z (_ BitVec 15))

  %concat = "smt.bv.concat"(%x, %y) : (!smt.bv.bv<5>, !smt.bv.bv<10>) -> !smt.bv.bv<15>
  %eq_concat = "smt.eq"(%z, %concat) : (!smt.bv.bv<15>, !smt.bv.bv<15>) -> !smt.bool
  "smt.assert"(%eq_concat) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (concat x y)))
}
