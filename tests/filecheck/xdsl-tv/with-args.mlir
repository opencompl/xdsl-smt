// RUN: xdsl-tv %s %S/with-args.mlir.output | z3 -in | filecheck "%s"

// This file uses `with-args.mlir.output`

builtin.module {
  func.func @test(%arg0 : i32) -> i32 {
    %x = arith.constant 3 : i32
    %r1 = arith.addi %arg0, %x : i32
    %y = arith.constant 4 : i32
    %r = arith.addi %r1, %y : i32
    func.return %r : i32
  }
}

// CHECK: unsat
