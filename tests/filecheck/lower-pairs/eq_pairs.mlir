// RUN: xdsl-smt "%s" -p=lower-pairs | filecheck "%s"

// Lower pairs from a "smt.eq" operation.

builtin.module {
  %0 = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>
  %1 = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>
  %eq = "smt.eq"(%0, %1) : (!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>, !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>) -> !smt.bool
  "smt.assert"(%eq) : (!smt.bool) -> ()
}

// CHECK:       %const_first = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:  %const_second_first = "smt.declare_const"() : () -> !smt.bv<32>
// CHECK-NEXT:  %const_second_second = "smt.declare_const"() : () -> !smt.bool
// CHECK:       %const_first_1 = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:  %const_second_first_1 = "smt.declare_const"() : () -> !smt.bv<32>
// CHECK-NEXT:  %const_second_second_1 = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:  %eq = "smt.eq"(%const_first, %const_first_1) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %eq_1 = "smt.eq"(%const_second_first, %const_second_first_1) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:  %eq_2 = "smt.eq"(%const_second_second, %const_second_second_1) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %eq_3 = "smt.and"(%eq_1, %eq_2) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %eq_4 = "smt.and"(%eq, %eq_3) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  "smt.assert"(%eq_4) : (!smt.bool) -> ()
