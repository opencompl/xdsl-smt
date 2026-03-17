// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8},lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8},lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.ssub_overflow"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer<1>
    "func.return"(%r) : (!transfer.integer<1>) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer<1>, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.bv<8>, %y : !smt.bv<8>, %1 : !effect.state):
// CHECK-NEXT:      %2 = "smt.bv.ssubo"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %3 = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK-NEXT:      %4 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:      %r = "smt.ite"(%2, %3, %4) : (!smt.bool, !smt.bv<1>, !smt.bv<1>) -> !smt.bv<1>
// CHECK-NEXT:      "smt.return"(%r, %1) : (!smt.bv<1>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.bv<8>, !smt.bv<8>, !effect.state) -> (!smt.bv<1>, !effect.state))
// CHECK-NEXT:  }
