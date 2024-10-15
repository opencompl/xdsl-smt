// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32):
    "func.return"(%x) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, %1 : !smt.bool):
// CHECK-NEXT:      %2 = "smt.utils.pair"(%x, %1) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%2) : (!smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>) -> ()
// CHECK-NEXT:    }) {"fun_name" = "test"} : () -> ((!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>)
// CHECK-NEXT:  }
