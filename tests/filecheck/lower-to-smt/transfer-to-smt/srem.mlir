// RUN: xdsl-smt %s -p=resolve-transfer-widths{width-map=\"default=8\"},lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=resolve-transfer-widths{width-map=\"default=8\"},lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer<@W>, %y : !transfer.integer<@W>):
    %r = "transfer.srem"(%x, %y) : (!transfer.integer<@W>,!transfer.integer<@W>) -> !transfer.integer<@W>
    "func.return"(%r) : (!transfer.integer<@W>) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer<@W>, !transfer.integer<@W>) -> !transfer.integer<@W>, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.bv<8>, %y : !smt.bv<8>, %1 : !effect.state):
// CHECK-NEXT:      %r = "smt.bv.srem"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      "smt.return"(%r, %1) : (!smt.bv<8>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.bv<8>, !smt.bv<8>, !effect.state) -> (!smt.bv<8>, !effect.state))
// CHECK-NEXT:  }
