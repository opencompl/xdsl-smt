// RUN: xdsl-smt "%s" -p=lower-abbv-to-bv | filecheck "%s"

builtin.module {
  %bvlhs = "test.op"() : () -> !smt.bv<32>
  %lhs = abbv.from_fixed_bitwidth %bvlhs : !smt.bv<32>
  %bvrhs = "test.op"() : () -> !smt.bv<32>
  %rhs = abbv.from_fixed_bitwidth %bvrhs : !smt.bv<32>
  %new_width = abbv.constant_bitwidth 64

  // CHECK:      %bvlhs = "test.op"() : () -> !smt.bv<32>
  // CHECK-NEXT: %bvrhs = "test.op"() : () -> !smt.bv<32>

  %concat = abbv.concat %lhs, %rhs
  %concat_bv = abbv.to_fixed_bitwidth %concat : !smt.bv<64>
  "test.op"(%concat_bv) : (!smt.bv<64>) -> ()
  // CHECK-NEXT: %concat_bv = "smt.bv.concat"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<64>
  // CHECK-NEXT: "test.op"(%concat_bv) : (!smt.bv<64>) -> ()

  %sext = abbv.sign_extend %lhs : %new_width
  %sext_bv = abbv.to_fixed_bitwidth %sext : !smt.bv<64>
  "test.op"(%sext_bv) : (!smt.bv<64>) -> ()
  // CHECK-NEXT: %sext_bv = "smt.bv.sign_extend"(%bvlhs) : (!smt.bv<32>) -> !smt.bv<64>
  // CHECK-NEXT: "test.op"(%sext_bv) : (!smt.bv<64>) -> ()

  %zext = abbv.zero_extend %rhs : %new_width
  %zext_bv = abbv.to_fixed_bitwidth %zext : !smt.bv<64>
  "test.op"(%zext_bv) : (!smt.bv<64>) -> ()
  // CHECK-NEXT: %zext_bv = "smt.bv.zero_extend"(%bvrhs) : (!smt.bv<32>) -> !smt.bv<64>
  // CHECK-NEXT: "test.op"(%zext_bv) : (!smt.bv<64>) -> ()
}
