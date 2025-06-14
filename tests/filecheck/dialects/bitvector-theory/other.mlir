// RUN: xdsl-smt "%s" | xdsl-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt "%s" -t=smt | z3 -in


builtin.module {
  %x = "smt.declare_const"() : () -> !smt.bv<5>
  %y = "smt.declare_const"() : () -> !smt.bv<10>
  %z = "smt.declare_const"() : () -> !smt.bv<15>
  // CHECK:      (declare-const $x (_ BitVec 5))
  // CHECK-NEXT: (declare-const $y (_ BitVec 10))
  // CHECK-NEXT: (declare-const $z (_ BitVec 15))

  %concat = "smt.bv.concat"(%x, %y) : (!smt.bv<5>, !smt.bv<10>) -> !smt.bv<15>
  %eq_concat = "smt.eq"(%z, %concat) : (!smt.bv<15>, !smt.bv<15>) -> !smt.bool
  "smt.assert"(%eq_concat) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $z (concat $x $y)))

  %extract = "smt.bv.extract"(%z) {start = #builtin.int<3>, end = #builtin.int<7>} : (!smt.bv<15>) -> !smt.bv<5>
  %eq_extract = "smt.eq"(%x, %extract) : (!smt.bv<5>, !smt.bv<5>) -> !smt.bool
  "smt.assert"(%eq_extract) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $x ((_ extract 7 3) $z))

  %repeat = "smt.bv.repeat"(%x) {"count" = #builtin.int<3>} : (!smt.bv<5>) -> !smt.bv<15>
  %eq_repeat = "smt.eq"(%z, %repeat) : (!smt.bv<15>, !smt.bv<15>) -> !smt.bool
  "smt.assert"(%eq_repeat) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $z ((_ repeat 3) $x))

  %zero_extend = "smt.bv.zero_extend"(%x) : (!smt.bv<5>) -> !smt.bv<15>
  %eq_zero_extend = "smt.eq"(%z, %zero_extend) : (!smt.bv<15>, !smt.bv<15>) -> !smt.bool
  "smt.assert"(%eq_zero_extend) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $z ((_ zero_extend 10) $x))

  %sign_extend = "smt.bv.sign_extend"(%x) : (!smt.bv<5>) -> !smt.bv<15>
  %eq_sign_extend = "smt.eq"(%z, %sign_extend) : (!smt.bv<15>, !smt.bv<15>) -> !smt.bool
  "smt.assert"(%eq_sign_extend) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $z ((_ sign_extend 10) $x))
}
