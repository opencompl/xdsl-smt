// XFAIL: *
// RUN: xdsl-smt "%s" -p=lower-to-smt -t=smt | filecheck "%s"

builtin.module {
    func.func private @eq(%cond : i1, %x : i32, %y : i32) -> i32 {
        %r = "llvm.select"(%cond, %x, %y) <{"fastmathFlags" = #llvm.fastmath<none>}> : (i1, i32, i32) -> i32
        func.return %r : i32
    }
}

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (define-fun eq ((cond (Pair (_ BitVec 1) Bool)) (x (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (let ((r (= (first cond) (_ bv1 1))))
// CHECK-NEXT:    (let ((tmp (second cond)))
// CHECK-NEXT:    (pair (ite r (first x) (first y)) (ite tmp tmp (ite r (second x) (second y)))))))
