// RUN: xdsl-smt "%s" -p=smt-extend | filecheck "%s"

// Lower pairs from a "smt.declare_const" operation.

builtin.module {
  %x = "smt.declare_const"() : () -> !smt.bv.bv<32>
  %y = "smt.declare_const"() : () -> !smt.bv.bv<32>
  %pred = "smt.bv.saddo"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool

  "smt.assert"(%pred) : (!smt.bool) -> ()
}

// CHECK:  %x = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:  %y = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:  %pred = "smt.bv.constant"() {value = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:  %pred_1 = "smt.bv.constant"() {value = #smt.bv.bv_val<2147483647: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:  %pred_2 = "smt.bv.constant"() {value = #smt.bv.bv_val<2147483648: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:  %pred_3 = "smt.bv.sgt"(%x, %pred) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:  %pred_4 = "smt.bv.sgt"(%y, %pred) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:  %pred_5 = "smt.bv.sub"(%pred_1, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:  %pred_6 = "smt.bv.sgt"(%x, %pred_5) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:  %pred_7 = "smt.and"(%pred_3, %pred_4) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %pred_8 = "smt.and"(%pred_7, %pred_6) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %pred_9 = "smt.bv.slt"(%x, %pred) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:  %pred_10 = "smt.bv.slt"(%y, %pred) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:  %pred_11 = "smt.bv.sub"(%pred_2, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:  %pred_12 = "smt.bv.slt"(%x, %pred_11) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:  %pred_13 = "smt.and"(%pred_9, %pred_10) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %pred_14 = "smt.and"(%pred_13, %pred_12) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %pred_15 = "smt.or"(%pred_8, %pred_14) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  "smt.assert"(%pred_15) : (!smt.bool) -> ()
