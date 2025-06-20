// RUN: xdsl-smt "%s" | xdsl-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt "%s" -t=smt | z3 -in


"builtin.module"() ({
  %x = "smt.declare_const"() : () -> !smt.bv<32>
  %y = "smt.declare_const"() : () -> !smt.bv<32>
  %z = "smt.declare_const"() : () -> !smt.bv<32>
  // CHECK:      (declare-const $x (_ BitVec 32))
  // CHECK-NEXT: (declare-const $y (_ BitVec 32))
  // CHECK-NEXT: (declare-const $z (_ BitVec 32))

  %not = "smt.bv.not"(%x) : (!smt.bv<32>) -> !smt.bv<32>
  %eq_not = "smt.eq"(%z, %not) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
  "smt.assert"(%eq_not) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $z (bvnot $x)))

  %or = "smt.bv.or"(%x, %y) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  %eq_or = "smt.eq"(%z, %or) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
  "smt.assert"(%eq_or) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $z (bvor $x $y)))

  %and = "smt.bv.and"(%x, %y) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  %eq_and = "smt.eq"(%z, %and) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
  "smt.assert"(%eq_and) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $z (bvand $x $y)))

  %xor = "smt.bv.xor"(%x, %y) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  %eq_xor = "smt.eq"(%z, %xor) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
  "smt.assert"(%eq_xor) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $z (bvxor $x $y)))

  %nand = "smt.bv.nand"(%x, %y) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  %eq_nand = "smt.eq"(%z, %nand) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
  "smt.assert"(%eq_nand) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $z (bvnand $x $y)))

  %nor = "smt.bv.nor"(%x, %y) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  %eq_nor = "smt.eq"(%z, %nor) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
  "smt.assert"(%eq_nor) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $z (bvnor $x $y)))

  %xnor = "smt.bv.xnor"(%x, %y) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  %eq_xnor = "smt.eq"(%z, %xnor) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
  "smt.assert"(%eq_xnor) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $z (bvxnor $x $y)))
}) : () -> ()
