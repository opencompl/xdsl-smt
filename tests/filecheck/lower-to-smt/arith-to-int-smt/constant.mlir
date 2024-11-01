// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects | filecheck %s
// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects -t=smt | z3 -in

%x = "arith.constant"() {"value" = 3 : i64} : () -> ui4

// CHECK:      builtin.module {
// CHECK-NEXT:  %0 = "smt.int.constant"() {"value" = 2 : i64} : () -> !smt.int.int
// CHECK-NEXT:  %1 = "smt.int.mul"(%0, %0) : (!smt.int.int, !smt.int.int) -> !smt.int.int
// CHECK-NEXT:  %2 = "smt.int.mul"(%1, %0) : (!smt.int.int, !smt.int.int) -> !smt.int.int
// CHECK-NEXT:  %3 = "smt.int.mul"(%2, %0) : (!smt.int.int, !smt.int.int) -> !smt.int.int
// CHECK-NEXT:  %4 = "smt.int.mod"(%5, %3) : (!smt.int.int, !smt.int.int) -> !smt.int.int
// CHECK-NEXT:  %5 = "smt.int.constant"() {"value" = 3 : i64} : () -> !smt.int.int
// CHECK-NEXT:  %6 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
// CHECK-NEXT:  %7 = "smt.utils.pair"(%4, %6) : (!smt.int.int, !smt.bool) -> !smt.utils.pair<!smt.int.int, !smt.bool>
// CHECK-NEXT:  %x = "smt.utils.pair"(%7, %3) : (!smt.utils.pair<!smt.int.int, !smt.bool>, !smt.int.int) -> !smt.utils.pair<!smt.utils.pair<!smt.int.int, !smt.bool>, !smt.int.int>
// CHECK-NEXT:}
