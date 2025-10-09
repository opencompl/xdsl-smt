// RUN: xdsl-tv %s %S/smt-to-arith.mlir.output -opt | z3 -in | filecheck "%s"

// This file uses `smt-to-arith.mlir.output`

builtin.module {
  func.func @test(%arg0: !smt.bv<8>, %arg1: !smt.bv<8>) -> !smt.bv<8> {
    %r = "smt.bv.udiv"(%arg0, %arg1) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
    func.return %r : !smt.bv<8>
  }
}

// CHECK: unsat
