// RUN: xdsl-smt "%s" | xdsl-smt | filecheck "%s"

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
// CHECK-NEXT:    %ub = ub.ub : !ub.ub_or<i32>
// CHECK-NEXT:    %non_ub = ub.from %value : !ub.ub_or<i32>
// CHECK-NEXT:    %res = ub.match %ub, %non_ub : (!ub.ub_or<i32>, !ub.ub_or<i32>) -> i64 {
// CHECK-NEXT:    ^0(%val1 : i32, %val2 : i32):
// CHECK-NEXT:      %x = "smt.declare_const"() : () -> i64
// CHECK-NEXT:      ub.yield %x : i64
// CHECK-NEXT:    } {
// CHECK-NEXT:      %y = "smt.declare_const"() : () -> i64
// CHECK-NEXT:      ub.yield %y : i64
// CHECK-NEXT:    }
// CHECK-NEXT:  }
