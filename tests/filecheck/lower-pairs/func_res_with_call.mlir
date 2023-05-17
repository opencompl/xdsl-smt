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


// CHECK:      %0 : !fun<[!smt.bool], [!smt.bool]> = smt.define_fun() ["fun_name" = "test_first"] {
// CHECK-NEXT: ^0(%1 : !smt.bool):
// CHECK-NEXT:   %2 : !smt.bool = smt.constant_bool false
// CHECK-NEXT:   smt.return(%2 : !smt.bool)
// CHECK-NEXT: }
// CHECK-NEXT: %3 : !fun<[!smt.bool], [!smt.bv.bv<32>]> = smt.define_fun() ["fun_name" = "test_second_first"] {
// CHECK-NEXT: ^1(%4 : !smt.bool):
// CHECK-NEXT:   %5 : !smt.bv.bv<32> = smt.bv.constant !smt.bv.bv_val<3: 32>
// CHECK-NEXT:   smt.return(%5 : !smt.bv.bv<32>)
// CHECK-NEXT: }
// CHECK-NEXT: %6 : !fun<[!smt.bool], [!smt.bool]> = smt.define_fun() ["fun_name" = "test_second_second"] {
// CHECK-NEXT: ^2(%arg : !smt.bool):
// CHECK-NEXT:   %7 : !smt.bool = smt.constant_bool false
// CHECK-NEXT:   smt.return(%7 : !smt.bool)
// CHECK-NEXT: }
// CHECK-NEXT: %true : !smt.bool = smt.constant_bool true
// CHECK-NEXT: %8 : !smt.bool = smt.call(%6 : !fun<[!smt.bool], [!smt.bool]>, %true : !smt.bool)
// CHECK-NEXT: smt.assert %8

