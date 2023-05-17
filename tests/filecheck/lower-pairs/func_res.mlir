// RUN: xdsl-smt.py %s -p=lower-pairs | filecheck %s

"builtin.module"() ({
  %0 = "smt.define_fun"() ({
    %1 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<3: 32>} : () -> !smt.bv.bv<32>
    %2 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
    %3 = "smt.utils.pair"(%1, %2) : (!smt.bv.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
    %4 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
    %5 = "smt.utils.pair"(%4, %3) : (!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>
    "smt.return"(%5) : (!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>) -> ()
  }) {"fun_name" = "test"} : () -> (() -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>)
}) : () -> ()


// CHECK:      %0 : !fun<[], [!smt.bool]> = smt.define_fun() ["fun_name" = "test_first"] {
// CHECK-NEXT:   %1 : !smt.bool = smt.constant_bool false
// CHECK-NEXT:   smt.return(%1 : !smt.bool)
// CHECK-NEXT: }
// CHECK-NEXT: %2 : !fun<[], [!smt.bv.bv<32>]> = smt.define_fun() ["fun_name" = "test_second_first"] {
// CHECK-NEXT:   %3 : !smt.bv.bv<32> = smt.bv.constant !smt.bv.bv_val<3: 32>
// CHECK-NEXT:   smt.return(%3 : !smt.bv.bv<32>)
// CHECK-NEXT: }
// CHECK-NEXT: %4 : !fun<[], [!smt.bool]> = smt.define_fun() ["fun_name" = "test_second_second"] {
// CHECK-NEXT:   %5 : !smt.bool = smt.constant_bool false
// CHECK-NEXT:   smt.return(%5 : !smt.bool)
// CHECK-NEXT: }

