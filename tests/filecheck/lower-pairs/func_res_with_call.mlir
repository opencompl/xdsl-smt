// RUN: xdsl-smt.py %s -p=lower-pairs | filecheck %s

// Check that we split functions returning pairs into multiple functions.

"builtin.module"() ({

  %0 = "smt.define_fun"() ({
  ^0(%arg : !smt.bool):
    %1 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<3: 32>} : () -> !smt.bv.bv<32>
    %2 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
    %3 = "smt.utils.pair"(%1, %2) : (!smt.bv.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
    %4 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
    %5 = "smt.utils.pair"(%4, %3) : (!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>
    "smt.return"(%5) : (!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>) -> ()
  }) {"fun_name" = "test"} : () -> ((!smt.bool) -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>)

  %true = "smt.constant_bool"() {"value" = #smt.bool_attr<true>} : () -> !smt.bool
  %6 = "smt.call"(%0, %true) : ((!smt.bool) -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>, !smt.bool) -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>
  %7 = "smt.utils.second"(%6) : (!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>) -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
  %8 = "smt.utils.second"(%7) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
  "smt.assert"(%8) : (!smt.bool) -> ()
}) : () -> ()


// CHECK:       %0 = "smt.define_fun"() ({
// CHECK-NEXT:  ^0(%arg : !smt.bool):
// CHECK-NEXT:    %1 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
// CHECK-NEXT:    "smt.return"(%1) : (!smt.bool) -> ()
// CHECK-NEXT:  }) {"fun_name" = "test_second_second"} : () -> ((!smt.bool) -> !smt.bool)
// CHECK-NEXT:  %2 = "smt.define_fun"() ({
// CHECK-NEXT:  ^1(%3 : !smt.bool):
// CHECK-NEXT:    %4 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<3: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:    "smt.return"(%4) : (!smt.bv.bv<32>) -> ()
// CHECK-NEXT:  }) {"fun_name" = "test_second_first"} : () -> ((!smt.bool) -> !smt.bv.bv<32>)
// CHECK-NEXT:  %5 = "smt.define_fun"() ({
// CHECK-NEXT:  ^2(%6 : !smt.bool):
// CHECK-NEXT:    %7 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
// CHECK-NEXT:    "smt.return"(%7) : (!smt.bool) -> ()
// CHECK-NEXT:  }) {"fun_name" = "test_first"} : () -> ((!smt.bool) -> !smt.bool)
// CHECK-NEXT:  %true = "smt.constant_bool"() {"value" = #smt.bool_attr<true>} : () -> !smt.bool
// CHECK-NEXT:  %8 = "smt.call"(%0, %true) : ((!smt.bool) -> !smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:  "smt.assert"(%8) : (!smt.bool) -> ()

