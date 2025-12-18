// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.usub_overflow"(%x, %y) : (!transfer.integer,!transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> i1, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "smt.define_fun"() ({
// CHECK-NEXT:   ^0(%x : !smt.bv<8>, %y : !smt.bv<8>, %1 : !effect.state):
// CHECK-NEXT:     %2 = "smt.bv.usubo"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:     %3 = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK-NEXT:     %4 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:     %5 = "smt.ite"(%2, %3, %4) : (!smt.bool, !smt.bv<1>, !smt.bv<1>) -> !smt.bv<1>
// CHECK-NEXT:     %6 = smt.constant false
// CHECK-NEXT:     %r = "smt.utils.pair"(%5, %6) : (!smt.bv<1>, !smt.bool) -> !smt.utils.pair<!smt.bv<1>, !smt.bool>
// CHECK-NEXT:     "smt.return"(%r, %1) : (!smt.utils.pair<!smt.bv<1>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:   }) {fun_name = "test"} : () -> ((!smt.bv<8>, !smt.bv<8>, !effect.state) -> (!smt.utils.pair<!smt.bv<1>, !smt.bool>, !effect.state))
// CHECK-NEXT: }
