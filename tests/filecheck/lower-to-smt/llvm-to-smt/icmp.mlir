// RUN: xdsl-smt "%s" -p=lower-to-smt -t=smt | filecheck "%s"

builtin.module {
    func.func private @eq(%x : i32, %y : i32) -> i1 {
        %r = llvm.icmp eq, %x, %y : i32
        func.return %r : i1
    }
// CHECK:       (define-fun eq ((x (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 1) Bool)
// CHECK-NEXT:    (pair (ite (= (first x) (first y)) (_ bv1 1) (_ bv0 1)) (or (second x) (second y))))

    func.func private @ne(%x : i32, %y : i32) -> i1 {
        %r = llvm.icmp ne, %x, %y : i32
        func.return %r : i1
    }
// CHECK-NEXT:  (define-fun ne ((x_0 (Pair (_ BitVec 32) Bool)) (y_0 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 1) Bool)
// CHECK-NEXT:    (pair (ite (distinct (first x_0) (first y_0)) (_ bv1 1) (_ bv0 1)) (or (second x_0) (second y_0))))

    func.func private @ugt(%x : i32, %y : i32) -> i1 {
        %r = llvm.icmp ugt, %x, %y : i32
        func.return %r : i1
    }
// CHECK-NEXT:  (define-fun ugt ((x_1 (Pair (_ BitVec 32) Bool)) (y_1 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 1) Bool)
// CHECK-NEXT:    (pair (ite (bvugt (first x_1) (first y_1)) (_ bv1 1) (_ bv0 1)) (or (second x_1) (second y_1))))

    func.func private @uge(%x : i32, %y : i32) -> i1 {
        %r = llvm.icmp uge, %x, %y : i32
        func.return %r : i1
    }
// CHECK-NEXT:  (define-fun uge ((x_2 (Pair (_ BitVec 32) Bool)) (y_2 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 1) Bool)
// CHECK-NEXT:    (pair (ite (bvuge (first x_2) (first y_2)) (_ bv1 1) (_ bv0 1)) (or (second x_2) (second y_2))))

    func.func private @ult(%x : i32, %y : i32) -> i1 {
        %r = llvm.icmp ult, %x, %y : i32
        func.return %r : i1
    }
// CHECK-NEXT:  (define-fun ult ((x_3 (Pair (_ BitVec 32) Bool)) (y_3 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 1) Bool)
// CHECK-NEXT:    (pair (ite (bvult (first x_3) (first y_3)) (_ bv1 1) (_ bv0 1)) (or (second x_3) (second y_3))))

    func.func private @ule(%x : i32, %y : i32) -> i1 {
        %r = llvm.icmp ule, %x, %y : i32
        func.return %r : i1
    }
// CHECK-NEXT:  (define-fun ule ((x_4 (Pair (_ BitVec 32) Bool)) (y_4 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 1) Bool)
// CHECK-NEXT:    (pair (ite (bvule (first x_4) (first y_4)) (_ bv1 1) (_ bv0 1)) (or (second x_4) (second y_4))))

    func.func private @sgt(%x : i32, %y : i32) -> i1 {
        %r = llvm.icmp sgt, %x, %y : i32
        func.return %r : i1
    }
// CHECK-NEXT:  (define-fun sgt ((x_5 (Pair (_ BitVec 32) Bool)) (y_5 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 1) Bool)
// CHECK-NEXT:    (pair (ite (bvsgt (first x_5) (first y_5)) (_ bv1 1) (_ bv0 1)) (or (second x_5) (second y_5))))

    func.func private @sge(%x : i32, %y : i32) -> i1 {
        %r = llvm.icmp sge, %x, %y : i32
        func.return %r : i1
    }
// CHECK-NEXT:  (define-fun sge ((x_6 (Pair (_ BitVec 32) Bool)) (y_6 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 1) Bool)
// CHECK-NEXT:    (pair (ite (bvsge (first x_6) (first y_6)) (_ bv1 1) (_ bv0 1)) (or (second x_6) (second y_6))))

    func.func private @slt(%x : i32, %y : i32) -> i1 {
        %r = llvm.icmp slt, %x, %y : i32
        func.return %r : i1
    }
// CHECK-NEXT:  (define-fun slt ((x_7 (Pair (_ BitVec 32) Bool)) (y_7 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 1) Bool)
// CHECK-NEXT:    (pair (ite (bvslt (first x_7) (first y_7)) (_ bv1 1) (_ bv0 1)) (or (second x_7) (second y_7))))

    func.func private @sle(%x : i32, %y : i32) -> i1 {
        %r = llvm.icmp sle, %x, %y : i32
        func.return %r : i1
    }
// CHECK-NEXT:  (define-fun sle ((x_8 (Pair (_ BitVec 32) Bool)) (y_8 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 1) Bool)
// CHECK-NEXT:    (pair (ite (bvsle (first x_8) (first y_8)) (_ bv1 1) (_ bv0 1)) (or (second x_8) (second y_8))))
}
