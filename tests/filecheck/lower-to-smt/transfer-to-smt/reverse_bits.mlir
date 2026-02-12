// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8},lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8},lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.reverse_bits"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer, "sym_visibility" = "private"} : () -> ()
}) : () -> ()


// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.bv<8>, %y : !smt.bv<8>, %1 : !effect.state):
// CHECK-NEXT:      %2 = "smt.bv.extract"(%x) {start = #builtin.int<0>, end = #builtin.int<0>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:      %3 = "smt.bv.extract"(%x) {start = #builtin.int<1>, end = #builtin.int<1>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:      %4 = "smt.bv.extract"(%x) {start = #builtin.int<2>, end = #builtin.int<2>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:      %5 = "smt.bv.extract"(%x) {start = #builtin.int<3>, end = #builtin.int<3>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:      %6 = "smt.bv.extract"(%x) {start = #builtin.int<4>, end = #builtin.int<4>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:      %7 = "smt.bv.extract"(%x) {start = #builtin.int<5>, end = #builtin.int<5>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:      %8 = "smt.bv.extract"(%x) {start = #builtin.int<6>, end = #builtin.int<6>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:      %9 = "smt.bv.extract"(%x) {start = #builtin.int<7>, end = #builtin.int<7>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:      %10 = "smt.bv.concat"(%2, %3) : (!smt.bv<1>, !smt.bv<1>) -> !smt.bv<2>
// CHECK-NEXT:      %11 = "smt.bv.concat"(%10, %4) : (!smt.bv<2>, !smt.bv<1>) -> !smt.bv<3>
// CHECK-NEXT:      %12 = "smt.bv.concat"(%11, %5) : (!smt.bv<3>, !smt.bv<1>) -> !smt.bv<4>
// CHECK-NEXT:      %13 = "smt.bv.concat"(%12, %6) : (!smt.bv<4>, !smt.bv<1>) -> !smt.bv<5>
// CHECK-NEXT:      %14 = "smt.bv.concat"(%13, %7) : (!smt.bv<5>, !smt.bv<1>) -> !smt.bv<6>
// CHECK-NEXT:      %15 = "smt.bv.concat"(%14, %8) : (!smt.bv<6>, !smt.bv<1>) -> !smt.bv<7>
// CHECK-NEXT:      %r = "smt.bv.concat"(%15, %9) : (!smt.bv<7>, !smt.bv<1>) -> !smt.bv<8>
// CHECK-NEXT:      "smt.return"(%r, %1) : (!smt.bv<8>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.bv<8>, !smt.bv<8>, !effect.state) -> (!smt.bv<8>, !effect.state))
// CHECK-NEXT:  }
