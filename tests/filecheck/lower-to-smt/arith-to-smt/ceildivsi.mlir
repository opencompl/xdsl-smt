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
// CHECK-NEXT:    ^0(%x : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %y : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %1 : !effect.state):
// CHECK-NEXT:      %2 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %3 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %4 = "smt.utils.first"(%y) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %5 = "smt.utils.second"(%y) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %6 = smt.bv.constant #smt.bv<2147483648> : !smt.bv<32>
// CHECK-NEXT:      %7 = smt.bv.constant #smt.bv<4294967295> : !smt.bv<32>
// CHECK-NEXT:      %8 = "smt.eq"(%2, %6) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:      %9 = "smt.eq"(%4, %7) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:      %10 = smt.and %8, %9
// CHECK-NEXT:      %11 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
// CHECK-NEXT:      %12 = "smt.eq"(%11, %4) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:      %13 = smt.or %10, %12
// CHECK-NEXT:      %14 = smt.or %13, %5
// CHECK-NEXT:      %15 = ub_effect.trigger %1
// CHECK-NEXT:      %16 = "smt.ite"(%14, %15, %1) : (!smt.bool, !effect.state, !effect.state) -> !effect.state
// CHECK-NEXT:      %17 = "smt.bv.sdiv"(%2, %4) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:      %18 = "smt.bv.mul"(%17, %4) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:      %19 = "smt.distinct"(%18, %2) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:      %20 = "smt.bv.sgt"(%2, %11) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:      %21 = "smt.bv.sgt"(%4, %11) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:      %22 = "smt.eq"(%20, %21) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %23 = smt.and %22, %19
// CHECK-NEXT:      %24 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
// CHECK-NEXT:      %25 = "smt.bv.add"(%17, %24) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:      %26 = "smt.ite"(%23, %25, %17) : (!smt.bool, !smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:      %r = "smt.utils.pair"(%26, %3) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r, %16) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state))
// CHECK-NEXT:  }
