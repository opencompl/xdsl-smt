// RUN: xdsl-smt "%s" -p=lower-abbv-to-bv | filecheck "%s"

builtin.module {
  %bv = "test.op"() : () -> !smt.bv<32>
  %abbv = abbv.from_fixed_bitwidth %bv : !smt.bv<32>
  // CHECK:      %bv = "test.op"() : () -> !smt.bv<32>

  %neg = abbv.neg %abbv
  "test.op"(%neg) : (!abbv.bv) -> ()
  // CHECK-NEXT: %0 = "smt.bv.neg"(%bv) : (!smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %neg = abbv.from_fixed_bitwidth %0 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%neg) : (!abbv.bv) -> ()


  %not = abbv.not %abbv
  "test.op"(%not) : (!abbv.bv) -> ()
  // CHECK-NEXT: %1 = "smt.bv.not"(%bv) : (!smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %not = abbv.from_fixed_bitwidth %1 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%not) : (!abbv.bv) -> ()
}
