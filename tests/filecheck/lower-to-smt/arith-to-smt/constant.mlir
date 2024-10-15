// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
    %x = "arith.constant"() {"value" = 3 : i32} : () -> i32
    "func.return"(%x) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = () -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()


// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%1 : !smt.bool):
// CHECK-NEXT:      %2 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<3: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:      %3 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
// CHECK-NEXT:      %x = "smt.utils.pair"(%2, %3) : (!smt.bv.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:      %4 = "smt.utils.pair"(%x, %1) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%4) : (!smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>) -> ()
// CHECK-NEXT:    }) {"fun_name" = "test"} : () -> ((!smt.bool) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.bool>)
// CHECK-NEXT:  }
