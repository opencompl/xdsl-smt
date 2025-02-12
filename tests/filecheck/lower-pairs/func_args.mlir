// RUN: xdsl-smt "%s" -p=lower-pairs | filecheck "%s"

// Check that pair function arguments are split into multiple arguments.

"builtin.module"() ({
  %0 = "smt.define_fun"() ({
  ^0(%arg : !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>):
    %1 = "smt.utils.second"(%arg) : (!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>) -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
    %2 = "smt.utils.first"(%1) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
    "smt.return"(%2) : (!smt.bv.bv<32>) -> ()
  }) {name = "test"} : () -> ((!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>) -> !smt.bv.bv<32>)
}) : () -> ()

// CHECK:      %0 = "smt.define_fun"() ({
// CHECK-NEXT: ^0(%1 : !smt.bool, %2 : !smt.bv.bv<32>, %3 : !smt.bool):
// CHECK-NEXT:   "smt.return"(%2) : (!smt.bv.bv<32>) -> ()
// CHECK-NEXT: }) {name = "test"} : () -> ((!smt.bool, !smt.bv.bv<32>, !smt.bool) -> !smt.bv.bv<32>)
