// RUN: xdsl-smt "%s" -p=smt-expand | filecheck "%s"

// Lower from a "smt.bv.umulo" operation.

builtin.module {
  %x = "smt.declare_const"() : () -> !smt.bv<32>
  %y = "smt.declare_const"() : () -> !smt.bv<32>
  %pred = "smt.bv.umulo"(%x, %y) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool

  "smt.assert"(%pred) : (!smt.bool) -> ()
}

// CHECK:       %x = "smt.declare_const"() : () -> !smt.bv<32>
// CHECK-NEXT:  %y = "smt.declare_const"() : () -> !smt.bv<32>
// CHECK-NEXT:  %pred = "smt.bv.umul_noovfl"(%x, %y) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:  %pred_1 = "smt.not"(%pred) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:  "smt.assert"(%pred_1) : (!smt.bool) -> ()
