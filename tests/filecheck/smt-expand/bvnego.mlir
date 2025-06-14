// RUN: xdsl-smt "%s" -p=smt-expand | filecheck "%s"

// Lower from a "smt.bv.nego" operation.

builtin.module {
  %x = "smt.declare_const"() : () -> !smt.bv<32>
  %y = "smt.declare_const"() : () -> !smt.bv<32>
  %pred = "smt.bv.nego"(%x) : (!smt.bv<32>) -> !smt.bool

  "smt.assert"(%pred) : (!smt.bool) -> ()
}

// CHECK:       %x = "smt.declare_const"() : () -> !smt.bv<32>
// CHECK-NEXT:  %y = "smt.declare_const"() : () -> !smt.bv<32>
// CHECK-NEXT:  %pred = "smt.bv.constant"() {value = #smt.bv.bv_val<2147483648: 32>} : () -> !smt.bv<32>
// CHECK-NEXT:  %pred_1 = "smt.eq"(%pred, %x) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:  "smt.assert"(%pred_1) : (!smt.bool) -> ()
