// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %neg = "transfer.is_all_ones"(%x) : (!transfer.integer) -> i1
    %r = "transfer.select"(%neg, %x, %y) : (i1, !transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "smt.define_fun"() ({
// CHECK-NEXT:   ^0(%x : !smt.bv<8>, %y : !smt.bv<8>, %1 : !effect.state):
// CHECK-NEXT:     %2 = smt.bv.constant #smt.bv<255> : !smt.bv<8>
// CHECK-NEXT:     %3 = "smt.eq"(%x, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:     %4 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:     %5 = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK-NEXT:     %6 = "smt.ite"(%3, %5, %4) : (!smt.bool, !smt.bv<1>, !smt.bv<1>) -> !smt.bv<1>
// CHECK-NEXT:     %7 = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK-NEXT:     %8 = "smt.eq"(%6, %7) : (!smt.bv<1>, !smt.bv<1>) -> !smt.bool
// CHECK-NEXT:     %r = "smt.ite"(%8, %x, %y) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:     "smt.return"(%r, %1) : (!smt.bv<8>, !effect.state) -> ()
// CHECK-NEXT:   }) {fun_name = "test"} : () -> ((!smt.bv<8>, !smt.bv<8>, !effect.state) -> (!smt.bv<8>, !effect.state))
// CHECK-NEXT: }
