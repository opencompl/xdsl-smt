// RUN: superoptimize %s --max-num-ops=1 --dialect=%S/arith.irdl | filecheck %s

func.func @foo(%x: !smt.bv<32>, %y: !smt.bv<32>) -> !smt.bv<32> {
  %r = "smt.bv.add"(%x, %y) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  func.return %r : !smt.bv<32>
}

// CHECK:       func.func @foo(%arg0 : i32, %arg1 : i32) -> i32 {
// CHECK-NEXT:    %0 = arith.addi %arg0, %arg1 : i32
// CHECK-NEXT:    func.return %0 : i32
// CHECK-NEXT:  }
