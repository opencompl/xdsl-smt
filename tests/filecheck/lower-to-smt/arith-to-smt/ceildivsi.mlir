// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.ceildivsi"(%x, %y) : (i32, i32) -> i32
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
// CHECK-NEXT:      %7 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<2147483648: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %8 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<4294967295: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %9 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<1: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %10 = "smt.eq"(%2, %7) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %11 = "smt.eq"(%4, %8) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %12 = "smt.and"(%10, %11) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %13 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %14 = "smt.eq"(%13, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %15 = "smt.or"(%12, %14) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %16 = "smt.bv.slt"(%2, %13) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %17 = "smt.ite"(%16, %8, %9) : (!smt.bool, !smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %18 = "smt.bv.sub"(%2, %17) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %19 = "smt.bv.sdiv"(%18, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %20 = "smt.bv.add"(%19, %9) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %21 = "smt.bv.sub"(%13, %2) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %22 = "smt.bv.sdiv"(%21, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %23 = "smt.bv.sub"(%13, %22) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %24 = "smt.bv.slt"(%4, %13) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %25 = "smt.bv.slt"(%13, %2) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %26 = "smt.bv.slt"(%13, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:      %27 = "smt.and"(%16, %24) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %28 = "smt.and"(%25, %26) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %29 = "smt.or"(%27, %28) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %30 = "smt.ite"(%29, %20, %23) : (!smt.bool, !smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %31 = "smt.or"(%15, %6) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %r = "smt.utils.pair"(%30, %31) : (!smt.bv.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:      %32 = "smt.utils.pair"(%r, %1) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%32) : (!smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>) -> ()
// CHECK-NEXT:    }) {"fun_name" = "test"} : () -> ((!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>)
// CHECK-NEXT:  }
