// RUN: xdsl-tv %S/with-args-input.mlir %S/with-args-output.mlir | filecheck "%s"

// This file uses `with-args-output.mlir`

builtin.module {
  func.func @test(%arg0 : i32) -> i32 {
    %x = arith.constant 3 : i32
    %r1 = arith.addi %arg0, %x : i32
    %y = arith.constant 4 : i32
    %r = arith.addi %r1, %y : i32
    func.return %r : i32
  }
}

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (define-fun test ((arg0 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (let ((y (pair (_ bv4 32) false)))
// CHECK-NEXT:    (let ((x (pair (_ bv3 32) false)))
// CHECK-NEXT:    (let ((r1 (pair (bvadd (first arg0) (first x)) (or (second arg0) (second x)))))
// CHECK-NEXT:    (pair (bvadd (first r1) (first y)) (or (second r1) (second y)))))))
// CHECK-NEXT:  (define-fun test_0 ((arg0_0 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (let ((x (pair (_ bv7 32) false)))
// CHECK-NEXT:    (pair (bvadd (first arg0_0) (first x)) (or (second arg0_0) (second x)))))
// CHECK-NEXT:  (declare-const tmp (Pair (_ BitVec 32) Bool))
// CHECK-NEXT:  (assert (not (= (test tmp) (test_0 tmp))))
// CHECK-NEXT:  (check-sat)
