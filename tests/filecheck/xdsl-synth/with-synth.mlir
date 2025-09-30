// RUN: xdsl-synth %s %S/with-synth.mlir.output | filecheck "%s"

// This file uses `with-synth.mlir.output`

builtin.module {
  func.func @test(%arg0 : i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %r = arith.muli %arg0, %c2 : i32
    func.return %r : i32
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    func.func @test(%arg0 : i32) -> i32 {
// CHECK-NEXT:      %c = arith.constant 1 : i32
// CHECK-NEXT:      %r = arith.shli %arg0, %c : i32
// CHECK-NEXT:      func.return %r : i32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
