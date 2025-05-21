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

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (assert (forall (($tmp (Pair (_ BitVec 32) Bool)) ($tmp_0 (Pair (Array Int (Pair (Array (_ BitVec 64) (Pair (_ BitVec 8) Bool))
// CHECK-NEXT:   (Pair (_ BitVec 64) Bool)))
// CHECK-NEXT:   Bool))) (let (($y (pair (_ bv4 32) false)))
// CHECK-NEXT:  (let (($x (pair (_ bv3 32) false)))
// CHECK-NEXT:  (let (($tmp_1 (pair false false)))
// CHECK-NEXT:  (let (($tmp_2 (first $tmp)))
// CHECK-NEXT:  (let (($tmp_3 (first $x)))
// CHECK-NEXT:  (let (($tmp_4 (_ bv0 32)))
// CHECK-NEXT:  (let (($tmp_5 (bvsge $tmp_2 $tmp_4)))
// CHECK-NEXT:  (let (($r1 (pair (bvadd $tmp_2 $tmp_3) (or (or (or false (and (and (= $tmp_5 (bvsge $tmp_3 $tmp_4)) (distinct $tmp_5 (bvsge (bvadd $tmp_2 $tmp_3) $tmp_4))) (second $tmp_1))) (and (bvult (bvadd $tmp_2 $tmp_3) $tmp_2) (first $tmp_1))) (or (second $tmp) (second $x))))))
// CHECK-NEXT:  (let (($tmp_6 (pair false false)))
// CHECK-NEXT:  (let (($tmp_7 (first $r1)))
// CHECK-NEXT:  (let (($tmp_8 (first $y)))
// CHECK-NEXT:  (let (($tmp_9 (_ bv0 32)))
// CHECK-NEXT:  (let (($tmp_10 (bvsge $tmp_7 $tmp_9)))
// CHECK-NEXT:  (let (($r (pair (bvadd $tmp_7 $tmp_8) (or (or (or false (and (and (= $tmp_10 (bvsge $tmp_8 $tmp_9)) (distinct $tmp_10 (bvsge (bvadd $tmp_7 $tmp_8) $tmp_9))) (second $tmp_6))) (and (bvult (bvadd $tmp_7 $tmp_8) $tmp_7) (first $tmp_6))) (or (second $r1) (second $y))))))
// CHECK-NEXT:  (let (($x_0 (pair (_ bv7 32) false)))
// CHECK-NEXT:  (let (($tmp_11 (pair false false)))
// CHECK-NEXT:  (let (($tmp_12 (first $tmp)))
// CHECK-NEXT:  (let (($tmp_13 (first $x_0)))
// CHECK-NEXT:  (let (($tmp_14 (_ bv0 32)))
// CHECK-NEXT:  (let (($tmp_15 (bvsge $tmp_12 $tmp_14)))
// CHECK-NEXT:  (let (($r_0 (pair (bvadd $tmp_12 $tmp_13) (or (or (or false (and (and (= $tmp_15 (bvsge $tmp_13 $tmp_14)) (distinct $tmp_15 (bvsge (bvadd $tmp_12 $tmp_13) $tmp_14))) (second $tmp_11))) (and (bvult (bvadd $tmp_12 $tmp_13) $tmp_12) (first $tmp_11))) (or (second $tmp) (second $x_0))))))
// CHECK-NEXT:  (or (and (not (second $tmp_0)) (and (and true (or (and (not (second $r_0)) (= (first $r) (first $r_0))) (second $r))) true)) (second $tmp_0)))))))))))))))))))))))))
// CHECK-NEXT:  (check-sat)
