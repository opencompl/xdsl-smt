// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.minsi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, %y : !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, %1 : !smt.bool):
// CHECK-NEXT:      %2 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %3 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %4 = "smt.utils.first"(%y) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %5 = "smt.utils.second"(%y) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %6 = "smt.or"(%3, %5) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %7 = "smt.bv.sle"(%2, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %8 = "smt.ite"(%7, %2, %4) : (!smt.bool, !smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %r = "smt.utils.pair"(%8, %6) : (!smt.bv.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:      %9 = "smt.utils.pair"(%r, %1) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%9) : (!smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>) -> ()
// CHECK-NEXT:    }) {"fun_name" = "test"} : () -> ((!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>)
// CHECK-NEXT:  }
