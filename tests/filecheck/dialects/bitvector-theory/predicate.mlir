// RUN: xdsl-smt "%s" | xdsl-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt "%s" -t=smt | z3 -in

"builtin.module"() ({
  %x = "smt.declare_const"() : () -> !smt.bv.bv<32>
  // CHECK:      (declare-const x (_ BitVec 32))
  %y = "smt.declare_const"() : () -> !smt.bv.bv<32>
  // CHECK-NEXT: (declare-const y (_ BitVec 32))

  %ule = "smt.bv.ule"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%ule) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (bvule x y))

  %ult = "smt.bv.ult"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%ult) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (bvult x y))

  %uge = "smt.bv.uge"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%uge) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (bvuge x y))

  %ugt = "smt.bv.ugt"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%ugt) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (bvugt x y))

  %sle = "smt.bv.sle"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%sle) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (bvsle x y))

  %slt = "smt.bv.slt"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%slt) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (bvslt x y))

  %sge = "smt.bv.sge"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%sge) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (bvsge x y))

  %sgt = "smt.bv.sgt"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%sgt) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (bvsgt x y))

  %umul_noovfl = "smt.bv.umul_noovfl"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%umul_noovfl) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (bvumul_noovfl x y))

  %smul_noovfl = "smt.bv.smul_noovfl"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%smul_noovfl) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (bvsmul_noovfl x y))

  %smul_noudfl = "smt.bv.smul_noudfl"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%smul_noudfl) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (bvsmul_noudfl x y))
}) : () -> ()
