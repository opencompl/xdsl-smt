// RUN: xdsl-smt "%s" -p=smt-expand | filecheck "%s"

// Lower from a "smt.bv.saddo" operation.

builtin.module {
  %x = "smt.declare_const"() : () -> !smt.bv.bv<32>
  %y = "smt.declare_const"() : () -> !smt.bv.bv<32>
  %pred = "smt.bv.saddo"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool

  "smt.assert"(%pred) : (!smt.bool) -> ()
}

// CHECK:  %x = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:  %y = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:  %pred = "smt.bv.constant"() {value = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:  %pred_1 = "smt.bv.add"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:  %pred_2 = "smt.bv.sge"(%x, %pred) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:  %pred_3 = "smt.bv.sge"(%y, %pred) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:  %pred_4 = "smt.bv.sge"(%pred_1, %pred) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:  %pred_5 = "smt.eq"(%pred_2, %pred_3) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %pred_6 = "smt.distinct"(%pred_2, %pred_4) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %pred_7 = "smt.and"(%pred_5, %pred_6) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  "smt.assert"(%pred_7) : (!smt.bool) -> ()
