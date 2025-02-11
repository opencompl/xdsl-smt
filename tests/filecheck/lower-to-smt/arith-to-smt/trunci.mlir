// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

builtin.module {
  func.func @test(%x : i32) -> i16 {
    %r = arith.trunci %x : i32 to i16
    "func.return"(%r) : (i16) -> ()
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, %1 : !effect.state):
// CHECK-NEXT:      %2 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %3 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %4 = "smt.bv.extract"(%2) {start = #builtin.int<0>, end = #builtin.int<15>} : (!smt.bv.bv<32>) -> !smt.bv.bv<16>
// CHECK-NEXT:      %r = "smt.utils.pair"(%4, %3) : (!smt.bv.bv<16>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<16>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%r, %1) : (!smt.utils.pair<!smt.bv.bv<16>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv.bv<16>, !smt.bool>, !effect.state))
// CHECK-NEXT:  }
