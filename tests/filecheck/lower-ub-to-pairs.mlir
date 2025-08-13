// RUN: xdsl-smt "%s" -p=lower-ub-to-pairs | filecheck "%s"

%value = "smt.declare_const"() : () -> i32
%ub = ub.ub : !ub.ub_or<i32>
%non_ub = ub.from %value : !ub.ub_or<i32>
%res = ub.match %ub, %non_ub : (!ub.ub_or<i32>, !ub.ub_or<i32>) -> i64 {
^bb0(%val1: i32, %val2: i32):
    %x = "smt.declare_const"() : () -> i64
    ub.yield %x : i64
} {
    %y = "smt.declare_const"() : () -> i64
    ub.yield %y : i64
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %value = "smt.declare_const"() : () -> i32
// CHECK-NEXT:    %0 = arith.constant 0 : i32
// CHECK-NEXT:    %ub = smt.constant true
// CHECK-NEXT:    %ub_1 = "smt.utils.pair"(%0, %ub) : (i32, !smt.bool) -> !smt.utils.pair<i32, !smt.bool>
// CHECK-NEXT:    %non_ub = smt.constant false
// CHECK-NEXT:    %non_ub_1 = "smt.utils.pair"(%value, %non_ub) : (i32, !smt.bool) -> !smt.utils.pair<i32, !smt.bool>
// CHECK-NEXT:    %val1 = "smt.utils.first"(%ub_1) : (!smt.utils.pair<i32, !smt.bool>) -> i32
// CHECK-NEXT:    %1 = "smt.utils.second"(%ub_1) : (!smt.utils.pair<i32, !smt.bool>) -> !smt.bool
// CHECK-NEXT:    %val2 = "smt.utils.first"(%non_ub_1) : (!smt.utils.pair<i32, !smt.bool>) -> i32
// CHECK-NEXT:    %2 = "smt.utils.second"(%non_ub_1) : (!smt.utils.pair<i32, !smt.bool>) -> !smt.bool
// CHECK-NEXT:    %3 = smt.or %2, %1
// CHECK-NEXT:    %x = "smt.declare_const"() : () -> i64
// CHECK-NEXT:    %y = "smt.declare_const"() : () -> i64
// CHECK-NEXT:    %res = "smt.ite"(%3, %y, %x) : (!smt.bool, i64, i64) -> i64
// CHECK-NEXT:  }
