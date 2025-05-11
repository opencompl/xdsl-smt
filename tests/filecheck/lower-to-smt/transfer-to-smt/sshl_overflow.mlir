// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,smt-expand,canonicalize,dce,merge-func-results -t=smt

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.sshl_overflow"(%x, %y) : (!transfer.integer,!transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> i1, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       %3 = "smt.bv.uge"(%y, %2) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bool
// CHECK:       %5 = "smt.bv.sge"(%x, %4) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bool
// CHECK:       %29 = "smt.bv.uge"(%y, %28) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bool
// CHECK:       %31 = "smt.bv.slt"(%x, %30) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bool
// CHECK:       %56 = "smt.bv.uge"(%y, %55) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bool
// CHECK-NEXT:    %57 = "smt.and"(%5, %29) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %58 = "smt.and"(%31, %56) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %59 = "smt.or"(%57, %58) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %60 = "smt.or"(%3, %59) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %61 = "smt.bv.constant"() {value = #smt.bv.bv_val<1: 1>} : () -> !smt.bv.bv<1>
// CHECK-NEXT:    %62 = "smt.bv.constant"() {value = #smt.bv.bv_val<0: 1>} : () -> !smt.bv.bv<1>
// CHECK-NEXT:    %63 = "smt.ite"(%60, %61, %62) : (!smt.bool, !smt.bv.bv<1>, !smt.bv.bv<1>) -> !smt.bv.bv<1>
// CHECK-NEXT:    %64 = "smt.constant_bool"() {value = #smt.bool_attr<false>} : () -> !smt.bool
// CHECK-NEXT:    %r = "smt.utils.pair"(%63, %64) : (!smt.bv.bv<1>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<1>, !smt.bool>
// CHECK-NEXT:    smt.return"(%r, %1) : (!smt.utils.pair<!smt.bv.bv<1>, !smt.bool>, !effect.state) -> ()
