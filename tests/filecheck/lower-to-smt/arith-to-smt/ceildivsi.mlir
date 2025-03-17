// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.ceildivsi"(%x, %y) : (i32, i32) -> i32
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
// CHECK-NEXT:      %6 = "smt.bv.constant"() {value = #smt.bv.bv_val<2147483648: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %7 = "smt.bv.constant"() {value = #smt.bv.bv_val<4294967295: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %8 = "smt.bv.constant"() {value = #smt.bv.bv_val<1: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %9 = "smt.eq"(%2, %6) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %10 = "smt.eq"(%4, %7) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %11 = "smt.and"(%9, %10) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %12 = "smt.bv.constant"() {value = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %13 = "smt.eq"(%12, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %14 = "smt.or"(%11, %13) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %15 = "smt.or"(%14, %5) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %16 = ub_effect.trigger %1
// CHECK-NEXT:      %17 = "smt.ite"(%15, %16, %1) : (!smt.bool, !effect.state, !effect.state) -> !effect.state
// CHECK-NEXT:      %18 = "smt.bv.slt"(%2, %12) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %19 = "smt.ite"(%18, %7, %8) : (!smt.bool, !smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %20 = "smt.bv.sub"(%2, %19) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %21 = "smt.bv.sdiv"(%20, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %22 = "smt.bv.add"(%21, %8) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %23 = "smt.bv.sub"(%12, %2) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %24 = "smt.bv.sdiv"(%23, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %25 = "smt.bv.sub"(%12, %24) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %26 = "smt.bv.slt"(%4, %12) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %27 = "smt.bv.slt"(%12, %2) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %28 = "smt.bv.slt"(%12, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %29 = "smt.and"(%18, %26) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %30 = "smt.and"(%27, %28) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %31 = "smt.or"(%29, %30) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %32 = "smt.ite"(%31, %22, %25) : (!smt.bool, !smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %r = "smt.utils.pair"(%32, %3) : (!smt.bv.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r, %17) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !effect.state))
// CHECK-NEXT:  }
