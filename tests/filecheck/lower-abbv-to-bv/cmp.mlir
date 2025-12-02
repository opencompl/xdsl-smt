// RUN: xdsl-smt "%s" -p=lower-abbv-to-bv | filecheck "%s"

builtin.module {
  %bvlhs = "test.op"() : () -> !smt.bv<32>
  %lhs = abbv.from_fixed_bitwidth %bvlhs : !smt.bv<32>

  %bvrhs = "test.op"() : () -> !smt.bv<32>
  %rhs = abbv.from_fixed_bitwidth %bvrhs : !smt.bv<32>

  %cmp = "abbv.cmp"(%lhs, %rhs) <{"pred" = 0 : i64}> : (!abbv.bv, !abbv.bv) -> !smt.bool
  "test.op"(%cmp) : (!smt.bool) -> ()
}

// CHECK:      %bvlhs = "test.op"() : () -> !smt.bv<32>
// CHECK-NEXT: %bvrhs = "test.op"() : () -> !smt.bv<32>
// CHECK-NEXT: %cmp = "smt.bv.cmp"(%bvlhs, %bvrhs) <{pred = 0 : i64}> : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT: "test.op"(%cmp) : (!smt.bool) -> ()
