// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize-smt -t=smt | z3 -in

builtin.module {
  func.func @test(%x : i32) -> i16 {
    %r = arith.trunci %x : i32 to i16
    "func.return"(%r) : (i16) -> ()
  }
}

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (define-fun test ((x (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 16) Bool)
// CHECK-NEXT:    (pair ((_ extract 15 0) (first x)) (second x)))
