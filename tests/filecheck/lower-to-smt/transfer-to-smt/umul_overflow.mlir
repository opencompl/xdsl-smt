// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,smt-expand,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.umul_overflow"(%x, %y) : (!transfer.integer,!transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> i1, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.bv.bv<8>, %y : !smt.bv.bv<8>, %1 : !effect.state):
// CHECK-NEXT:      %r = "smt.bv.umul_noovfl"(%x, %y) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bool
// CHECK-NEXT:      %r_1 = "smt.bv.constant"() {value = #smt.bv.bv_val<0: 1>} : () -> !smt.bv.bv<1>
// CHECK-NEXT:      %r_2 = "smt.bv.constant"() {value = #smt.bv.bv_val<1: 1>} : () -> !smt.bv.bv<1>
// CHECK-NEXT:      %r_3 = "smt.ite"(%r, %r_2, %r_1) : (!smt.bool, !smt.bv.bv<1>, !smt.bv.bv<1>) -> !smt.bv.bv<1>
// CHECK-NEXT:      %r_4 = "smt.constant_bool"() {value = #smt.bool_attr<false>} : () -> !smt.bool
// CHECK-NEXT:      %r_5 = "smt.utils.pair"(%r_3, %r_4) : (!smt.bv.bv<1>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<1>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r_5, %1) : (!smt.utils.pair<!smt.bv.bv<1>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.bv.bv<8>, !smt.bv.bv<8>, !effect.state) -> (!smt.utils.pair<!smt.bv.bv<1>, !smt.bool>, !effect.state))
// CHECK-NEXT:  }
