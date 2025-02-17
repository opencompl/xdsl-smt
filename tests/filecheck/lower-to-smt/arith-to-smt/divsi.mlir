// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.divsi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, %y : !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, %1 : !effect.state):
// CHECK-NEXT:      %2 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %3 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %4 = "smt.utils.first"(%y) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %5 = "smt.utils.second"(%y) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %6 = "smt.or"(%3, %5) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %7 = "smt.bv.constant"() {value = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %8 = "smt.eq"(%4, %7) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %9 = "smt.bv.constant"() {value = #smt.bv.bv_val<2147483648: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %10 = "smt.bv.constant"() {value = #smt.bv.bv_val<4294967295: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %11 = "smt.eq"(%2, %9) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %12 = "smt.eq"(%4, %10) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %13 = "smt.and"(%11, %12) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %14 = ub_effect.trigger %1
// CHECK-NEXT:      %15 = "smt.ite"(%8, %14, %1) : (!smt.bool, !effect.state, !effect.state) -> !effect.state
// CHECK-NEXT:      %16 = "smt.bv.sdiv"(%2, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %17 = "smt.or"(%13, %6) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %r = "smt.utils.pair"(%16, %17) : (!smt.bv.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r, %15) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !effect.state))
// CHECK-NEXT:  }
