// RUN: xdsl-smt "%s" -p=lower-pairs | filecheck "%s"

// Check that we split functions returning pairs into multiple functions.

"builtin.module"() ({

  %0 = "smt.define_fun"() ({
  ^0(%arg : !smt.bool):
    %1 = smt.bv.constant #smt.bv<3> : !smt.bv<32>
    %2 = smt.constant false
    %3 = "smt.utils.pair"(%1, %2) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
    %4 = smt.constant false
    %5 = "smt.utils.pair"(%4, %3) : (!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>
    "smt.return"(%5) : (!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>) -> ()
  }) {fun_name = "test"} : () -> ((!smt.bool) -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>)

  %true = smt.constant true
  %6 = "smt.call"(%0, %true) : ((!smt.bool) -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>, !smt.bool) -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>
  %7 = "smt.utils.second"(%6) : (!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv<32>, !smt.bool>>) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
  %8 = "smt.utils.second"(%7) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
  "smt.assert"(%8) : (!smt.bool) -> ()
}) : () -> ()


// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%arg : !smt.bool):
// CHECK-NEXT:      %1 = smt.constant false
// CHECK-NEXT:      "smt.return"(%1) : (!smt.bool) -> ()
// CHECK-NEXT:    }) {fun_name = "test_second_second"} : () -> ((!smt.bool) -> !smt.bool)
// CHECK-NEXT:    %2 = "smt.define_fun"() ({
// CHECK-NEXT:    ^1(%arg_1 : !smt.bool):
// CHECK-NEXT:      %3 = smt.bv.constant #smt.bv<3> : !smt.bv<32>
// CHECK-NEXT:      "smt.return"(%3) : (!smt.bv<32>) -> ()
// CHECK-NEXT:    }) {fun_name = "test_second_first"} : () -> ((!smt.bool) -> !smt.bv<32>)
// CHECK-NEXT:    %4 = "smt.define_fun"() ({
// CHECK-NEXT:    ^2(%arg_2 : !smt.bool):
// CHECK-NEXT:      %5 = smt.constant false
// CHECK-NEXT:      "smt.return"(%5) : (!smt.bool) -> ()
// CHECK-NEXT:    }) {fun_name = "test_first"} : () -> ((!smt.bool) -> !smt.bool)
// CHECK-NEXT:    %true = smt.constant true
// CHECK-NEXT:    %6 = "smt.call"(%0, %true) : ((!smt.bool) -> !smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    "smt.assert"(%6) : (!smt.bool) -> ()
// CHECK-NEXT:  }
