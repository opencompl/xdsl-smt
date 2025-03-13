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
// CHECK-NEXT:  (define-fun $test (($arg0 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (let (($c2 (pair (_ bv2 32) false)))
// CHECK-NEXT:    (pair (bvmul (first $arg0) (first $c2)) (or (second $arg0) (second $c2)))))
// CHECK-NEXT:  (define-fun $test_0 (($arg0_0 (Pair (_ BitVec 32) Bool)) ($c (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (let (($tmp (first $c)))
// CHECK-NEXT:    (pair (bvshl (first $arg0_0) $tmp) (or (bvugt $tmp (_ bv32 32)) (or (second $arg0_0) (second $c))))))
// CHECK-NEXT:  (declare-const $tmp (_ BitVec 32))
// CHECK-NEXT:  (assert (forall (($tmp_0 (Pair (_ BitVec 32) Bool))) (let (($tmp_1 ($test_0 $tmp_0 (pair $tmp false))))
// CHECK-NEXT:  (let (($tmp_2 ($test $tmp_0)))
// CHECK-NEXT:  (or (and (not (second $tmp_1)) (= (first $tmp_2) (first $tmp_1))) (second $tmp_2))))))
// CHECK-NEXT:  (check-sat)

