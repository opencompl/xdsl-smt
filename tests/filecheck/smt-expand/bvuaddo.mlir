// RUN: xdsl-smt "%s" -p=smt-expand | filecheck "%s"

// Lower pairs from a "smt.declare_const" operation.

builtin.module {
  %x = "smt.declare_const"() : () -> !smt.bv.bv<32>
  %y = "smt.declare_const"() : () -> !smt.bv.bv<32>
  %pred = "smt.bv.uaddo"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool

  "smt.assert"(%pred) : (!smt.bool) -> ()
}

// CHECK:       %x = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:  %y = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:  %pred = "smt.bv.add"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:  %pred_1 = "smt.bv.ult"(%pred, %x) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:  "smt.assert"(%pred_1) : (!smt.bool) -> ()
