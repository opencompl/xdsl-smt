// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %low, %high = arith.mului_extended %x, %y : i32
    func.return %low, %high : i32, i32
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> (i32, i32), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %y : !smt.utils.pair<!smt.bv<32>, !smt.bool>, %1 : !effect.state):
// CHECK-NEXT:      %2 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %3 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %4 = "smt.utils.first"(%y) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:      %5 = "smt.utils.second"(%y) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %6 = "smt.or"(%3, %5) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %7 = "smt.bv.zero_extend"(%2) : (!smt.bv<32>) -> !smt.bv<64>
// CHECK-NEXT:      %8 = "smt.bv.zero_extend"(%4) : (!smt.bv<32>) -> !smt.bv<64>
// CHECK-NEXT:      %9 = "smt.bv.mul"(%7, %8) : (!smt.bv<64>, !smt.bv<64>) -> !smt.bv<64>
// CHECK-NEXT:      %10 = "smt.bv.extract"(%9) {start = #builtin.int<0>, end = #builtin.int<31>} : (!smt.bv<64>) -> !smt.bv<32>
// CHECK-NEXT:      %11 = "smt.bv.extract"(%9) {start = #builtin.int<32>, end = #builtin.int<63>} : (!smt.bv<64>) -> !smt.bv<32>
// CHECK-NEXT:      %low = "smt.utils.pair"(%10, %6) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:      %high = "smt.utils.pair"(%11, %6) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%low, %high, %1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv<32>, !smt.bool>, !effect.state))
// CHECK-NEXT:  }
