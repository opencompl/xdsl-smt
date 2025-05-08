// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.srem"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.bv.bv<8>, %y : !smt.bv.bv<8>, %1 : !effect.state):
// CHECK-NEXT:      %r = "smt.bv.srem"(%x, %y) : (!smt.bv.bv<8>, !smt.bv.bv<8>) -> !smt.bv.bv<8>
// CHECK-NEXT:      "smt.return"(%r, %1) : (!smt.bv.bv<8>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.bv.bv<8>, !smt.bv.bv<8>, !effect.state) -> (!smt.bv.bv<8>, !effect.state))
// CHECK-NEXT:  }
