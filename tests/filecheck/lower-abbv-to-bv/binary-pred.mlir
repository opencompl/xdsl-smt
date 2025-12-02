// RUN: xdsl-smt "%s" -p=lower-abbv-to-bv | filecheck "%s"

builtin.module {
  %bvlhs = "test.op"() : () -> !smt.bv<32>
  %lhs = abbv.from_fixed_bitwidth %bvlhs : !smt.bv<32>
  // CHECK:      %bvlhs = "test.op"() : () -> !smt.bv<32>
  // CHECK-NEXT: %lhs = abbv.from_fixed_bitwidth %bvlhs : !smt.bv<32>

  %bvrhs = "test.op"() : () -> !smt.bv<32>
  %rhs = abbv.from_fixed_bitwidth %bvrhs : !smt.bv<32>
  // CHECK-NEXT: %bvrhs = "test.op"() : () -> !smt.bv<32>
  // CHECK-NEXT: %rhs = abbv.from_fixed_bitwidth %bvrhs : !smt.bv<32>

  %uadd = abbv.uaddo %lhs, %rhs
  "test.op"(%uadd) : (!smt.bool) -> ()
  // CHECK-NEXT: %uadd = abbv.uaddo %lhs, %rhs
  // CHECK-NEXT: "test.op"(%uadd) : (!smt.bool) -> ()

  %sadd = abbv.saddo %lhs, %rhs
  "test.op"(%sadd) : (!smt.bool) -> ()
  // CHECK-NEXT: %sadd = abbv.saddo %lhs, %rhs
  // CHECK-NEXT: "test.op"(%sadd) : (!smt.bool) -> ()

  %umul = abbv.umulo %lhs, %rhs
  "test.op"(%umul) : (!smt.bool) -> ()
  // CHECK-NEXT: %umul = abbv.umulo %lhs, %rhs
  // CHECK-NEXT: "test.op"(%umul) : (!smt.bool) -> ()

  %smul = abbv.smulo %lhs, %rhs
  "test.op"(%smul) : (!smt.bool) -> ()
  // CHECK-NEXT: %smul = abbv.smulo %lhs, %rhs
  // CHECK-NEXT: "test.op"(%smul) : (!smt.bool) -> ()

  %umul_no = abbv.umul_noovfl %lhs, %rhs
  "test.op"(%umul_no) : (!smt.bool) -> ()
  // CHECK-NEXT: %umul_no = abbv.umul_noovfl %lhs, %rhs
  // CHECK-NEXT: "test.op"(%umul_no) : (!smt.bool) -> ()

  %smul_no = abbv.smul_noovfl %lhs, %rhs
  "test.op"(%smul_no) : (!smt.bool) -> ()
  // CHECK-NEXT: %smul_no = abbv.smul_noovfl %lhs, %rhs
  // CHECK-NEXT: "test.op"(%smul_no) : (!smt.bool) -> ()

  %smul_nu = abbv.smul_noudfl %lhs, %rhs
  "test.op"(%smul_nu) : (!smt.bool) -> ()
  // CHECK-NEXT: %smul_nu = abbv.smul_noudfl %lhs, %rhs
  // CHECK-NEXT: "test.op"(%smul_nu) : (!smt.bool) -> ()
}
