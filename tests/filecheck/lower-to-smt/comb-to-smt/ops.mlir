// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

// comb.add
"func.func"() ({
^0(%x: i32):
  %r = "comb.add"(%x) : (i32) -> i32
  "func.return"(%r) : (i32) -> ()
}) {"sym_name" = "add_one", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK:       %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %1 : !effect.state):
// CHECK-NEXT:      %2 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %3 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %r = "smt.utils.pair"(%2, %3) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r, %1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "add_one"} : () -> ((!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state))


"func.func"() ({
^0(%x: i32, %y: i32):
  %r = "comb.add"(%x, %y) : (i32, i32) -> i32
  "func.return"(%r) : (i32) -> ()
}) {"sym_name" = "add_two", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK:         %4 = "smt.define_fun"() ({
// CHECK-NEXT:    ^1(%x_1 : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %y : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %5 : !effect.state):
// CHECK-NEXT:      %6 = "smt.utils.first"(%x_1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %7 = "smt.utils.second"(%x_1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %8 = "smt.utils.first"(%y) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %9 = "smt.utils.second"(%y) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %10 = "smt.or"(%7, %9) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %11 = "smt.bv.add"(%6, %8) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:      %r_1 = "smt.utils.pair"(%11, %10) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r_1, %5) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "add_two"} : () -> ((!smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state))

"func.func"() ({
^0(%x: i32, %y: i32, %z: i32):
  %r = "comb.add"(%x, %y, %z) : (i32, i32, i32) -> i32
  "func.return"(%r) : (i32) -> ()
}) {"sym_name" = "add_three", "function_type" = (i32, i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK:         %12 = "smt.define_fun"() ({
// CHECK-NEXT:    ^2(%x_2 : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %y_1 : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %z : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %13 : !effect.state):
// CHECK-NEXT:      %14 = "smt.utils.first"(%x_2) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %15 = "smt.utils.second"(%x_2) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %16 = "smt.utils.first"(%y_1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %17 = "smt.utils.second"(%y_1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %18 = "smt.or"(%15, %17) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %19 = "smt.utils.first"(%z) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %20 = "smt.utils.second"(%z) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %21 = "smt.or"(%18, %20) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %22 = "smt.bv.add"(%14, %16) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:      %23 = "smt.bv.add"(%22, %19) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:      %r_2 = "smt.utils.pair"(%23, %21) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r_2, %13) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "add_three"} : () -> ((!smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state))

// -----

// comb.mul

"func.func"() ({
^0(%x: i32):
  %r = "comb.mul"(%x) : (i32) -> i32
  "func.return"(%r) : (i32) -> ()
}) {"sym_name" = "mul_one", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK:         %24 = "smt.define_fun"() ({
// CHECK-NEXT:    ^3(%x_3 : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %25 : !effect.state):
// CHECK-NEXT:      %26 = "smt.utils.first"(%x_3) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %27 = "smt.utils.second"(%x_3) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %r_3 = "smt.utils.pair"(%26, %27) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r_3, %25) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "mul_one"} : () -> ((!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state))


"func.func"() ({
^0(%x: i32, %y: i32):
  %r = "comb.mul"(%x, %y) : (i32, i32) -> i32
  "func.return"(%r) : (i32) -> ()
}) {"sym_name" = "mul_two", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK:         %28 = "smt.define_fun"() ({
// CHECK-NEXT:    ^4(%x_4 : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %y_2 : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %29 : !effect.state):
// CHECK-NEXT:      %30 = "smt.utils.first"(%x_4) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %31 = "smt.utils.second"(%x_4) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %32 = "smt.utils.first"(%y_2) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %33 = "smt.utils.second"(%y_2) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %34 = "smt.or"(%31, %33) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %35 = "smt.bv.mul"(%30, %32) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:      %r_4 = "smt.utils.pair"(%35, %34) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r_4, %29) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "mul_two"} : () -> ((!smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state))

"func.func"() ({
^0(%x: i32, %y: i32, %z: i32):
  %r = "comb.mul"(%x, %y, %z) : (i32, i32, i32) -> i32
  "func.return"(%r) : (i32) -> ()
}) {"sym_name" = "mul_three", "function_type" = (i32, i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()

// CHECK:         %36 = "smt.define_fun"() ({
// CHECK-NEXT:    ^5(%x_5 : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %y_3 : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %z_1 : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %37 : !effect.state):
// CHECK-NEXT:      %38 = "smt.utils.first"(%x_5) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %39 = "smt.utils.second"(%x_5) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %40 = "smt.utils.first"(%y_3) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %41 = "smt.utils.second"(%y_3) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %42 = "smt.or"(%39, %41) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %43 = "smt.utils.first"(%z_1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %44 = "smt.utils.second"(%z_1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %45 = "smt.or"(%42, %44) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %46 = "smt.bv.mul"(%38, %40) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:      %47 = "smt.bv.mul"(%46, %43) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:      %r_5 = "smt.utils.pair"(%47, %45) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r_5, %37) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "mul_three"} : () -> ((!smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state))
