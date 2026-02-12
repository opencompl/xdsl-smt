// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8},lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8},lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer):
    %r = "transfer.popcount"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer) -> !transfer.integer, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "smt.define_fun"() ({
// CHECK-NEXT:   ^0(%x : !smt.bv<8>, %1 : !effect.state):
// CHECK-NEXT:     %2 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:     %3 = "smt.bv.extract"(%x) {start = #builtin.int<0>, end = #builtin.int<0>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:     %4 = "smt.bv.zero_extend"(%3) : (!smt.bv<1>) -> !smt.bv<8>
// CHECK-NEXT:     %5 = "smt.bv.add"(%2, %4) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:     %6 = "smt.bv.extract"(%x) {start = #builtin.int<1>, end = #builtin.int<1>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:     %7 = "smt.bv.zero_extend"(%6) : (!smt.bv<1>) -> !smt.bv<8>
// CHECK-NEXT:     %8 = "smt.bv.add"(%5, %7) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:     %9 = "smt.bv.extract"(%x) {start = #builtin.int<2>, end = #builtin.int<2>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:     %10 = "smt.bv.zero_extend"(%9) : (!smt.bv<1>) -> !smt.bv<8>
// CHECK-NEXT:     %11 = "smt.bv.add"(%8, %10) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:     %12 = "smt.bv.extract"(%x) {start = #builtin.int<3>, end = #builtin.int<3>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:     %13 = "smt.bv.zero_extend"(%12) : (!smt.bv<1>) -> !smt.bv<8>
// CHECK-NEXT:     %14 = "smt.bv.add"(%11, %13) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:     %15 = "smt.bv.extract"(%x) {start = #builtin.int<4>, end = #builtin.int<4>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:     %16 = "smt.bv.zero_extend"(%15) : (!smt.bv<1>) -> !smt.bv<8>
// CHECK-NEXT:     %17 = "smt.bv.add"(%14, %16) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:     %18 = "smt.bv.extract"(%x) {start = #builtin.int<5>, end = #builtin.int<5>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:     %19 = "smt.bv.zero_extend"(%18) : (!smt.bv<1>) -> !smt.bv<8>
// CHECK-NEXT:     %20 = "smt.bv.add"(%17, %19) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:     %21 = "smt.bv.extract"(%x) {start = #builtin.int<6>, end = #builtin.int<6>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:     %22 = "smt.bv.zero_extend"(%21) : (!smt.bv<1>) -> !smt.bv<8>
// CHECK-NEXT:     %23 = "smt.bv.add"(%20, %22) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:     %24 = "smt.bv.extract"(%x) {start = #builtin.int<7>, end = #builtin.int<7>} : (!smt.bv<8>) -> !smt.bv<1>
// CHECK-NEXT:     %25 = "smt.bv.zero_extend"(%24) : (!smt.bv<1>) -> !smt.bv<8>
// CHECK-NEXT:     %r = "smt.bv.add"(%23, %25) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:     "smt.return"(%r, %1) : (!smt.bv<8>, !effect.state) -> ()
// CHECK-NEXT:   }) {fun_name = "test"} : () -> ((!smt.bv<8>, !effect.state) -> (!smt.bv<8>, !effect.state))
// CHECK-NEXT: }
