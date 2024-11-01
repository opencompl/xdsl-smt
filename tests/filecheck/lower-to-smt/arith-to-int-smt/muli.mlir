// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects | filecheck %s
// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects -t=smt | z3 -in

%x = "arith.constant"() {"value" = 3 : i64} : () -> ui4
%y = "arith.muli"(%x,%x): (ui4,ui4) -> ui4

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
// CHECK-NEXT:  %8 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.utils.pair<!smt.int.int, !smt.bool>, !smt.int.int>) -> !smt.utils.pair<!smt.int.int, !smt.bool>
// CHECK-NEXT:  %9 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.utils.pair<!smt.int.int, !smt.bool>, !smt.int.int>) -> !smt.utils.pair<!smt.int.int, !smt.bool>
// CHECK-NEXT:  %10 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.utils.pair<!smt.int.int, !smt.bool>, !smt.int.int>) -> !smt.int.int
// CHECK-NEXT:  %11 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.utils.pair<!smt.int.int, !smt.bool>, !smt.int.int>) -> !smt.int.int
// CHECK-NEXT:  %12 = "smt.eq"(%10, %11) : (!smt.int.int, !smt.int.int) -> !smt.bool
// CHECK-NEXT:  "smt.assert"(%12) : (!smt.bool) -> ()
// CHECK-NEXT:  %13 = "smt.utils.first"(%8) : (!smt.utils.pair<!smt.int.int, !smt.bool>) -> !smt.int.int
// CHECK-NEXT:  %14 = "smt.utils.first"(%9) : (!smt.utils.pair<!smt.int.int, !smt.bool>) -> !smt.int.int
// CHECK-NEXT:  %15 = "smt.utils.second"(%8) : (!smt.utils.pair<!smt.int.int, !smt.bool>) -> !smt.bool
// CHECK-NEXT:  %16 = "smt.utils.second"(%9) : (!smt.utils.pair<!smt.int.int, !smt.bool>) -> !smt.bool
// CHECK-NEXT:  %17 = "smt.int.mul"(%13, %14) : (!smt.int.int, !smt.int.int) -> !smt.int.int
// CHECK-NEXT:  %18 = "smt.int.mod"(%17, %10) : (!smt.int.int, !smt.int.int) -> !smt.int.int
// CHECK-NEXT:  %19 = "smt.or"(%15, %16) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  %20 = "smt.utils.pair"(%18, %19) : (!smt.int.int, !smt.bool) -> !smt.utils.pair<!smt.int.int, !smt.bool>
// CHECK-NEXT:  %y = "smt.utils.pair"(%20, %10) : (!smt.utils.pair<!smt.int.int, !smt.bool>, !smt.int.int) -> !smt.utils.pair<!smt.utils.pair<!smt.int.int, !smt.bool>, !smt.int.int>
// CHECK-NEXT:}
