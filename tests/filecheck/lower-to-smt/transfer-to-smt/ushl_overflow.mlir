// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,smt-expand,canonicalize,dce,merge-func-results -t=smt

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.ushl_overflow"(%x, %y) : (!transfer.integer,!transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> i1, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       %3 = "smt.bv.uge"(%y, %2) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bool
// CHECK:       %27 = "smt.bv.ugt"(%y, %26) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bool
// CHECK-NEXT:  %28 = "smt.or"(%3, %27) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %29 = "smt.bv.constant"() {value = #smt.bv.bv_val<1: 1>} : () -> !smt.bv.bv<1>
// CHECK-NEXT:  %30 = "smt.bv.constant"() {value = #smt.bv.bv_val<0: 1>} : () -> !smt.bv.bv<1>
// CHECK-NEXT:  %31 = "smt.ite"(%28, %29, %30) : (!smt.bool, !smt.bv.bv<1>, !smt.bv.bv<1>) -> !smt.bv.bv<1>
// CHECK-NEXT:  "smt.constant_bool"() {value = #smt.bool_attr<false>} : () -> !smt.bool
// CHECK-NEXT:  %r = "smt.utils.pair"(%31, %32) : (!smt.bv.bv<1>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<1>, !smt.bool>
// CHECK-NEXT:  "smt.return"(%r, %1) : (!smt.utils.pair<!smt.bv.bv<1>, !smt.bool>, !effect.state) -> ()
