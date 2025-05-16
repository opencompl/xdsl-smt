// RUN: xdsl-synth %s %S/without-synth.mlir.output -opt | filecheck "%s"

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
// CHECK-NEXT:  (define-fun $test_second (($tmp (_ BitVec 32)) ($tmp_0 Bool)) Bool
// CHECK-NEXT:    $tmp_0)
// CHECK-NEXT:  (define-fun $test_first (($tmp_1 (_ BitVec 32)) ($tmp_2 Bool)) (_ BitVec 32)
// CHECK-NEXT:    (bvadd (bvadd $tmp_1 (_ bv3 32)) (_ bv4 32)))
// CHECK-NEXT:  (define-fun $test_second_0 (($tmp_3 (_ BitVec 32)) ($tmp_4 Bool)) Bool
// CHECK-NEXT:    $tmp_4)
// CHECK-NEXT:  (define-fun $test_first_0 (($tmp_5 (_ BitVec 32)) ($tmp_6 Bool)) (_ BitVec 32)
// CHECK-NEXT:    (bvadd $tmp_5 (_ bv7 32)))
// CHECK-NEXT:  (assert (forall (($tmp_7 (_ BitVec 32)) ($tmp_8 Bool)) (or (and (not ($test_second_0 $tmp_7 $tmp_8)) (= ($test_first $tmp_7 $tmp_8) ($test_first_0 $tmp_7 $tmp_8))) ($test_second $tmp_7 $tmp_8))))
// CHECK-NEXT:  (check-sat)
