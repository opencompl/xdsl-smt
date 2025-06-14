// RUN: xdsl-smt "%s" -p=lower-pairs | filecheck "%s"

// Lower pairs from a "smt.distinct" operation.

builtin.module {
  %0 = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>
  %1 = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>
  %distinct = "smt.distinct"(%0, %1) : (!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>, !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>) -> !smt.bool
  "smt.assert"(%distinct) : (!smt.bool) -> ()
}

// CHECK:       %const_first = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:  %const_second_first = "smt.declare_const"() : () -> !smt.bv<32>
// CHECK-NEXT:  %const_second_second = "smt.declare_const"() : () -> !smt.bool
// CHECK:       %const_first_1 = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:  %const_second_first_1 = "smt.declare_const"() : () -> !smt.bv<32>
// CHECK-NEXT:  %const_second_second_1 = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:  %distinct = "smt.distinct"(%const_first, %const_first_1) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %distinct_1 = "smt.distinct"(%const_second_first, %const_second_first_1) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:  %distinct_2 = "smt.distinct"(%const_second_second, %const_second_second_1) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %distinct_3 = "smt.or"(%distinct_1, %distinct_2) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %distinct_4 = "smt.or"(%distinct, %distinct_3) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  "smt.assert"(%distinct_4) : (!smt.bool) -> ()
