// RUN: xdsl-smt "%s" -p=lower-pairs | filecheck "%s"

"builtin.module"() ({
  %0 = "smt.define_fun"() ({
    %1 = smt.bv.constant #smt.bv<3> : !smt.bv<32>
    %2 = smt.constant false
    %3 = "smt.utils.pair"(%1, %2) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
    %4 = smt.constant false
    %5 = "smt.utils.pair"(%4, %3) : (!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>
    "smt.return"(%5) : (!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>) -> ()
  }) {fun_name = "test"} : () -> (() -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>)
}) : () -> ()


// CHECK:      %0 = "smt.define_fun"() ({
// CHECK-NEXT:   %1 = smt.constant false
// CHECK-NEXT:   "smt.return"(%1) : (!smt.bool) -> ()
// CHECK-NEXT: }) {fun_name = "test_second_second"} : () -> (() -> !smt.bool)
// CHECK-NEXT: %2 = "smt.define_fun"() ({
// CHECK-NEXT:   %3 = smt.bv.constant #smt.bv<3> : !smt.bv<32>
// CHECK-NEXT:   "smt.return"(%3) : (!smt.bv<32>) -> ()
// CHECK-NEXT: }) {fun_name = "test_second_first"} : () -> (() -> !smt.bv<32>)
// CHECK-NEXT: %4 = "smt.define_fun"() ({
// CHECK-NEXT:   %5 = smt.constant false
// CHECK-NEXT:   "smt.return"(%5) : (!smt.bool) -> ()
// CHECK-NEXT: }) {fun_name = "test_first"} : () -> (() -> !smt.bool)
