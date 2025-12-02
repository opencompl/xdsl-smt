// RUN: xdsl-smt "%s" -p=lower-abbv-to-bv | filecheck "%s"

builtin.module {
  %1 = "test.op"() : () -> !smt.bv<32>
  %2 = abbv.from_fixed_bitwidth %1 : !smt.bv<32>
  %3 = abbv.get_bitwidth %2
  "test.op"(%3) : (!abbv.bitwidth) -> ()
}

// CHECK:      %0 = "test.op"() : () -> !smt.bv<32>
// CHECK-NEXT: %1 = abbv.constant_bitwidth 32 {value = #builtin.int<32>}
// CHECK-NEXT: "test.op"(%1) : (!abbv.bitwidth) -> ()
