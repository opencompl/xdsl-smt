// RUN: xdsl-smt "%s" -p=lower-abbv-to-bv | filecheck "%s"

builtin.module {
  %1 = "test.op"() : () -> !smt.bv<32>
  %2 = abbv.from_fixed_bitwidth %1 : !smt.bv<32>
  %3 = abbv.to_fixed_bitwidth %2 : !smt.bv<32>
  "test.op"(%3) : (!smt.bv<32>) -> ()
}

// CHECK:      %0 = "test.op"() : () -> !smt.bv<32>
// CHECK-NEXT: "test.op"(%0) : (!smt.bv<32>) -> ()
