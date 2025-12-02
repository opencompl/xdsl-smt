// RUN: xdsl-smt "%s" -p=lower-abbv-to-bv | filecheck "%s"

builtin.module {
  %bvlhs = "test.op"() : () -> !smt.bv<32>
  %lhs = abbv.from_fixed_bitwidth %bvlhs : !smt.bv<32>
  // CHECK: %bvlhs = "test.op"() : () -> !smt.bv<32>

  %bvrhs = "test.op"() : () -> !smt.bv<32>
  %rhs = abbv.from_fixed_bitwidth %bvrhs : !smt.bv<32>
  // CHECK: %bvrhs = "test.op"() : () -> !smt.bv<32>

  %add = abbv.add %lhs, %rhs
  "test.op"(%add) : (!abbv.bv) -> ()
  // CHECK-NEXT: %0 = "smt.bv.add"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %add = abbv.from_fixed_bitwidth %0 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%add) : (!abbv.bv) -> ()

  %sub = abbv.sub %lhs, %rhs
  "test.op"(%sub) : (!abbv.bv) -> ()
  // CHECK-NEXT: %1 = "smt.bv.sub"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %sub = abbv.from_fixed_bitwidth %1 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%sub) : (!abbv.bv) -> ()

  %mul = abbv.mul %lhs, %rhs
  "test.op"(%mul) : (!abbv.bv) -> ()
  // CHECK-NEXT: %2 = "smt.bv.mul"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %mul = abbv.from_fixed_bitwidth %2 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%mul) : (!abbv.bv) -> ()

  %urem = abbv.urem %lhs, %rhs
  "test.op"(%urem) : (!abbv.bv) -> ()
  // CHECK-NEXT: %3 = "smt.bv.urem"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %urem = abbv.from_fixed_bitwidth %3 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%urem) : (!abbv.bv) -> ()

  %srem = abbv.srem %lhs, %rhs
  "test.op"(%srem) : (!abbv.bv) -> ()
  // CHECK-NEXT: %4 = "smt.bv.srem"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %srem = abbv.from_fixed_bitwidth %4 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%srem) : (!abbv.bv) -> ()

  %smod = abbv.smod %lhs, %rhs
  "test.op"(%smod) : (!abbv.bv) -> ()
  // CHECK-NEXT: %5 = "smt.bv.smod"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %smod = abbv.from_fixed_bitwidth %5 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%smod) : (!abbv.bv) -> ()

  %shl = abbv.shl %lhs, %rhs
  "test.op"(%shl) : (!abbv.bv) -> ()
  // CHECK-NEXT: %6 = "smt.bv.shl"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %shl = abbv.from_fixed_bitwidth %6 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%shl) : (!abbv.bv) -> ()

  %lshr = abbv.lshr %lhs, %rhs
  "test.op"(%lshr) : (!abbv.bv) -> ()
  // CHECK-NEXT: %7 = "smt.bv.lshr"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %lshr = abbv.from_fixed_bitwidth %7 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%lshr) : (!abbv.bv) -> ()

  %ashr = abbv.ashr %lhs, %rhs
  "test.op"(%ashr) : (!abbv.bv) -> ()
  // CHECK-NEXT: %8 = "smt.bv.ashr"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %ashr = abbv.from_fixed_bitwidth %8 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%ashr) : (!abbv.bv) -> ()

  %udiv = abbv.udiv %lhs, %rhs
  "test.op"(%udiv) : (!abbv.bv) -> ()
  // CHECK-NEXT: %9 = "smt.bv.udiv"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %udiv = abbv.from_fixed_bitwidth %9 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%udiv) : (!abbv.bv) -> ()

  %sdiv = abbv.sdiv %lhs, %rhs
  "test.op"(%sdiv) : (!abbv.bv) -> ()
  // CHECK-NEXT: %10 = "smt.bv.sdiv"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %sdiv = abbv.from_fixed_bitwidth %10 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%sdiv) : (!abbv.bv) -> ()

  %or = abbv.or %lhs, %rhs
  "test.op"(%or) : (!abbv.bv) -> ()
  // CHECK-NEXT: %11 = "smt.bv.or"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %or = abbv.from_fixed_bitwidth %11 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%or) : (!abbv.bv) -> ()

  %xor = abbv.xor %lhs, %rhs
  "test.op"(%xor) : (!abbv.bv) -> ()
  // CHECK-NEXT: %12 = "smt.bv.xor"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %xor = abbv.from_fixed_bitwidth %12 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%xor) : (!abbv.bv) -> ()

  %and = abbv.and %lhs, %rhs
  "test.op"(%and) : (!abbv.bv) -> ()
  // CHECK-NEXT: %13 = "smt.bv.and"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %and = abbv.from_fixed_bitwidth %13 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%and) : (!abbv.bv) -> ()

  %nand = abbv.nand %lhs, %rhs
  "test.op"(%nand) : (!abbv.bv) -> ()
  // CHECK-NEXT: %14 = "smt.bv.nand"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %nand = abbv.from_fixed_bitwidth %14 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%nand) : (!abbv.bv) -> ()

  %nor = abbv.nor %lhs, %rhs
  "test.op"(%nor) : (!abbv.bv) -> ()
  // CHECK-NEXT: %15 = "smt.bv.nor"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %nor = abbv.from_fixed_bitwidth %15 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%nor) : (!abbv.bv) -> ()

  %xnor = abbv.xnor %lhs, %rhs
  "test.op"(%xnor) : (!abbv.bv) -> ()
  // CHECK-NEXT: %16 = "smt.bv.xnor"(%bvlhs, %bvrhs) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  // CHECK-NEXT: %xnor = abbv.from_fixed_bitwidth %16 : !smt.bv<32>
  // CHECK-NEXT: "test.op"(%xnor) : (!abbv.bv) -> ()
}
