// RUN: xdsl-smt "%s" -p=lower-abbv-to-bv | filecheck "%s"

builtin.module {
  %0 = abbv.constant_bitwidth 32
  %1 = abbv.constant 42 : %0
  "test.op"(%1) : (!abbv.bv) -> ()
}

// CHECK:      %0 = smt.bv.constant #smt.bv<42> : !smt.bv<32>
// CHECK-NEXT: %1 = abbv.from_fixed_bitwidth %0 : !smt.bv<32>
// CHECK-NEXT: "test.op"(%1) : (!abbv.bv) -> ()
