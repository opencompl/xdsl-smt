// RUN: xdsl-synth %s %S/with-synth.mlir.output | filecheck "%s"

// This file uses `with-synth.mlir.output`

builtin.module {
  func.func @test(%arg0 : i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %r = arith.muli %arg0, %c2 : i32
    func.return %r : i32
  }
}

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (declare-const $c (Pair (_ BitVec 32) Bool))
// CHECK-NEXT:  (assert (forall (($tmp (Pair (_ BitVec 32) Bool)) ($tmp_0 (Pair (Array Int (Pair (Array (_ BitVec 64) (Pair (_ BitVec 8) Bool))
// CHECK-NEXT:   (Pair (_ BitVec 64) Bool)))
// CHECK-NEXT:   Bool))) (let (($c2 (pair (_ bv2 32) false)))
// CHECK-NEXT:  (let (($r (pair (bvmul (first $tmp) (first $c2)) (or (second $tmp) (second $c2)))))
// CHECK-NEXT:  (let (($tmp_1 (first $c)))
// CHECK-NEXT:  (let (($r_0 (pair (bvshl (first $tmp) $tmp_1) (or (bvugt $tmp_1 (_ bv32 32)) (or (second $tmp) (second $c))))))
// CHECK-NEXT:  (or (and (not (second $tmp_0)) (and (and true (or (and (not (second $r_0)) (= (first $r) (first $r_0))) (second $r))) true)) (second $tmp_0))))))))
// CHECK-NEXT:  (check-sat)
