// RUN: xdsl-smt "%s" -p=lower-pairs | filecheck "%s"

// Lower pairs from a "smt.declare_const" operation.

builtin.module {
  %0 = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>
  %1 = "smt.utils.first"(%0) : (!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>) -> !smt.bool
  %2 = "smt.utils.second"(%0) : (!smt.utils.pair<!smt.bool, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>>) -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
  %3 = "smt.utils.first"(%2) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
  %4 = "smt.utils.second"(%2) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
  %func = "smt.define_fun"() ({
    ^0(%arg0: !smt.bool, %arg1: !smt.bv.bv<32>, %arg2: !smt.bool):
        "smt.return"(%arg0) : (!smt.bool) -> ()
  }) {"name" = "test"} : () -> ((!smt.bool, !smt.bv.bv<32>, !smt.bool) -> (!smt.bool))
  %func_res = "smt.call"(%func, %1, %3, %4) : ((!smt.bool, !smt.bv.bv<32>, !smt.bool) -> (!smt.bool), !smt.bool, !smt.bv.bv<32>, !smt.bool) -> !smt.bool
  "smt.assert"(%func_res) : (!smt.bool) -> ()
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %1 = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:    %2 = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %func = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%arg0 : !smt.bool, %arg1 : !smt.bv.bv<32>, %arg2 : !smt.bool):
// CHECK-NEXT:      "smt.return"(%arg0) : (!smt.bool) -> ()
// CHECK-NEXT:    }) {"name" = "test"} : () -> ((!smt.bool, !smt.bv.bv<32>, !smt.bool) -> !smt.bool)
// CHECK-NEXT:    %func_res = "smt.call"(%func, %0, %1, %2) : ((!smt.bool, !smt.bv.bv<32>, !smt.bool) -> !smt.bool, !smt.bool, !smt.bv.bv<32>, !smt.bool) -> !smt.bool
// CHECK-NEXT:    "smt.assert"(%func_res) : (!smt.bool) -> ()
// CHECK-NEXT:  }
