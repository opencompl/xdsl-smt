// RUN: xdsl-synth %s %S/without-synth.mlir.output | filecheck "%s"

// This file uses `without-synth.mlir.output`

builtin.module {
  func.func @test(%arg0 : i32) -> i32 {
    %x = arith.constant 3 : i32
    %r1 = arith.addi %arg0, %x : i32
    %y = arith.constant 4 : i32
    %r = arith.addi %r1, %y : i32
    func.return %r : i32
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @test(%arg0 : i32) -> i32 {
// CHECK-NEXT:      %x = arith.constant 7 : i32
// CHECK-NEXT:      %r = arith.addi %arg0, %x : i32
// CHECK-NEXT:      func.return %r : i32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
