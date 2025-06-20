// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.cmpi"(%x, %y) {"predicate" = 5 : i64} : (i32, i32) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i1, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %y : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %1 : !effect.state):
// CHECK-NEXT:      %2 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %3 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %4 = "smt.utils.first"(%y) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %5 = "smt.utils.second"(%y) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %6 = smt.or %3, %5
// CHECK-NEXT:      %7 = "smt.bv.sge"(%2, %4) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:      %8 = "smt.bv.constant"() {value = #smt.bv<0> : !smt.bv<1>} : () -> !smt.bv<1>
// CHECK-NEXT:      %9 = "smt.bv.constant"() {value = #smt.bv<1> : !smt.bv<1>} : () -> !smt.bv<1>
// CHECK-NEXT:      %10 = "smt.ite"(%7, %9, %8) : (!smt.bool, !smt.bv<1>, !smt.bv<1>) -> !smt.bv<1>
// CHECK-NEXT:      %r = "smt.utils.pair"(%10, %6) : (!smt.bv<1>, !smt.bool) -> !smt.utils.pair<!smt.bv<1>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r, %1) : (!smt.utils.pair<!smt.bv<1>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv<1>, !smt.bool>, !effect.state))
// CHECK-NEXT:  }
