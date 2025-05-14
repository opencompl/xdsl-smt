// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.clear_low_bits"(%x, %y) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:	builtin.module {
// CHECK-NEXT:  %0 = "smt.define_fun"() ({
// CHECK-NEXT:  ^0(%x : !smt.bv.bv<8>, %y : !smt.bv.bv<8>, %1 : !effect.state):
// CHECK-NEXT:    %2 = "smt.bv.constant"() {value = #smt.bv.bv_val<0: 8>} : () -> !smt.bv.bv<8>
// CHECK-NEXT:    %3 = "smt.bv.constant"() {value = #smt.bv.bv_val<8: 8>} : () -> !smt.bv.bv<8>
// CHECK-NEXT:    %4 = "smt.bv.sub"(%3, %y) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bv.bv<8>
// CHECK-NEXT:    %5 = "smt.bv.constant"() {value = #smt.bv.bv_val<255: 8>} : () -> !smt.bv.bv<8>
// CHECK-NEXT:    %6 = "smt.bv.ashr"(%5, %2) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bv.bv<8>
// CHECK-NEXT:    %7 = "smt.bv.shl"(%6, %2) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bv.bv<8>
// CHECK-NEXT:    %8 = "smt.bv.shl"(%7, %4) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bv.bv<8>
// CHECK-NEXT:    %9 = "smt.bv.lshr"(%8, %4) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bv.bv<8>
// CHECK-NEXT:    %10 = "smt.bv.not"(%9) : (!smt.bv.bv<8>) -> !smt.bv.bv<8>
// CHECK-NEXT:    %r = "smt.bv.and"(%10, %x) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bv.bv<8>
// CHECK-NEXT:    "smt.return"(%r, %1) : (!smt.bv.bv<8>, !effect.state) -> ()
// CHECK-NEXT:  }) {fun_name = "test"} : () -> ((!smt.bv.bv<8>, !smt.bv.bv<8>, !effect.state) -> (!smt.bv.bv<8>, !effect.state))
// CHECK-NEXT:}
