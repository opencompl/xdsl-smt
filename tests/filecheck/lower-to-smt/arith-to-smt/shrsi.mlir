// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.shrsi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %y : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %1 : !effect.state):
// CHECK-NEXT:      %2 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %3 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %4 = "smt.utils.first"(%y) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %5 = "smt.utils.second"(%y) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %6 = smt.or %3, %5
// CHECK-NEXT:      %7 = smt.bv.constant #smt.bv<32> : !smt.bv<32>
// CHECK-NEXT:      %8 = "smt.bv.ugt"(%4, %7) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:      %9 = "smt.bv.ashr"(%2, %4) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:      %10 = smt.or %8, %6
// CHECK-NEXT:      %r = "smt.utils.pair"(%9, %10) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r, %1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state))
// CHECK-NEXT:  }
