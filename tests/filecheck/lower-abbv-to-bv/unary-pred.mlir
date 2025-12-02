// RUN: xdsl-smt "%s" -p=lower-abbv-to-bv | filecheck "%s"

builtin.module {
  %bv = "test.op"() : () -> !smt.bv<32>
  %abbv = abbv.from_fixed_bitwidth %bv : !smt.bv<32>

  %neg = abbv.nego %abbv
  "test.op"(%neg) : (!smt.bool) -> ()
}

// CHECK:      %bv = "test.op"() : () -> !smt.bv<32>
// CHECK-NEXT: %neg = "smt.bv.nego"(%bv) : (!smt.bv<32>) -> !smt.bool
// CHECK-NEXT: "test.op"(%neg) : (!smt.bool) -> ()
