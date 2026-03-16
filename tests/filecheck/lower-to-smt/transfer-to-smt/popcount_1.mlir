// RUN: xdsl-smt %s -w 1 -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -w 1 -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.popcount"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer) -> !transfer.integer, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "smt.define_fun"() ({
// CHECK-NEXT:   ^0(%x : !smt.bv<1>, %1 : !effect.state):
// CHECK-NEXT:     %2 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:     %3 = "smt.bv.extract"(%x) {start = #builtin.int<0>, end = #builtin.int<0>} : (!smt.bv<1>) -> !smt.bv<1>
// CHECK-NEXT:     %r = "smt.bv.add"(%2, %3) : (!smt.bv<1>, !smt.bv<1>) -> !smt.bv<1>
// CHECK-NEXT:     "smt.return"(%r, %1) : (!smt.bv<1>, !effect.state) -> ()
// CHECK-NEXT:   }) {fun_name = "test"} : () -> ((!smt.bv<1>, !effect.state) -> (!smt.bv<1>, !effect.state))
// CHECK-NEXT: }
