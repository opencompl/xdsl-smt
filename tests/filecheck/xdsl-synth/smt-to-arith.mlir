// RUN: xdsl-synth %s %S/smt-to-arith.mlir.output | filecheck "%s"

// This file uses `smt-to-arith.mlir.output`

func.func @test(%arg0: !smt.bv<8>, %arg1: !smt.bv<8>) -> !smt.bv<8> {
  %r = "smt.bv.udiv"(%arg0, %arg1) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  func.return %r : !smt.bv<8>
}

// CHECK:         func.func @test(%arg0 : i8, %arg1 : i8) -> i8 {
// CHECK-NEXT:      %zero = arith.constant 0 : i8
// CHECK-NEXT:      %is_zero_div = arith.cmpi eq, %arg1, %zero : i8
// CHECK-NEXT:      %minus_one = arith.constant -1 : i8
// CHECK-NEXT:      %one = arith.constant 1 : i8
// CHECK-NEXT:      %lhs = arith.select %is_zero_div, %minus_one, %arg0 : i8
// CHECK-NEXT:      %rhs = arith.select %is_zero_div, %one, %arg1 : i8
// CHECK-NEXT:      %r = arith.divui %lhs, %rhs : i8
// CHECK-NEXT:      func.return %r : i8
// CHECK-NEXT:    }
