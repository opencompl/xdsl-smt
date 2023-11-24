// RUN: xdsl-smt "%s" -p=lower-to-smt -t=smt | filecheck "%s"

builtin.module {
    func.func private @add(%x : i32, %y : i32) -> i32 {
        %r = llvm.add %x, %y : i32
        func.return %r : i32
    }
    // CHECK:       (define-fun add ((x (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (pair (bvadd (first x) (first y)) (or (second x) (second y))))

    func.func private @sub(%x : i32, %y : i32) -> i32 {
        %r = llvm.sub %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun sub ((x_0 (Pair (_ BitVec 32) Bool)) (y_0 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (pair (bvsub (first x_0) (first y_0)) (or (second x_0) (second y_0))))

    func.func private @mul(%x : i32, %y : i32) -> i32 {
        %r = llvm.mul %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun mul ((x_1 (Pair (_ BitVec 32) Bool)) (y_1 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (pair (bvmul (first x_1) (first y_1)) (or (second x_1) (second y_1))))

    func.func private @udiv(%x : i32, %y : i32) -> i32 {
        %r = llvm.udiv %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun udiv ((x_2 (Pair (_ BitVec 32) Bool)) (y_2 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (let ((tmp (first y_2)))
    // CHECK-NEXT:    (pair (bvudiv (first x_2) tmp) (or (= tmp (_ bv0 32)) (or (second x_2) (second y_2))))))

    func.func private @sdiv(%x : i32, %y : i32) -> i32 {
        %r = llvm.sdiv %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun sdiv ((x_3 (Pair (_ BitVec 32) Bool)) (y_3 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (let ((tmp (first y_3)))
    // CHECK-NEXT:    (let ((tmp_0 (first x_3)))
    // CHECK-NEXT:    (pair (bvsdiv tmp_0 tmp) (or (or (= (_ bv0 32) tmp) (and (= tmp_0 (_ bv2147483648 32)) (= tmp (_ bv4294967295 32)))) (or (second x_3) (second y_3)))))))

    func.func private @urem(%x : i32, %y : i32) -> i32 {
        %r = llvm.urem %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun urem ((x_4 (Pair (_ BitVec 32) Bool)) (y_4 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (let ((tmp (first y_4)))
    // CHECK-NEXT:    (pair (bvurem (first x_4) tmp) (or (= tmp (_ bv0 32)) (or (second x_4) (second y_4))))))

    func.func private @srem(%x : i32, %y : i32) -> i32 {
        %r = llvm.srem %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun srem ((x_5 (Pair (_ BitVec 32) Bool)) (y_5 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (let ((tmp (first y_5)))
    // CHECK-NEXT:    (pair (bvsrem (first x_5) tmp) (or (= tmp (_ bv0 32)) (or (second x_5) (second y_5))))))

    func.func private @and(%x : i32, %y : i32) -> i32 {
        %r = llvm.and %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun and ((x_6 (Pair (_ BitVec 32) Bool)) (y_6 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (pair (bvand (first x_6) (first y_6)) (or (second x_6) (second y_6))))

    func.func private @or(%x : i32, %y : i32) -> i32 {
        %r = llvm.or %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun or ((x_7 (Pair (_ BitVec 32) Bool)) (y_7 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (pair (bvor (first x_7) (first y_7)) (or (second x_7) (second y_7))))

    func.func private @xor(%x : i32, %y : i32) -> i32 {
        %r = llvm.xor %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun xor ((x_8 (Pair (_ BitVec 32) Bool)) (y_8 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (pair (bvxor (first x_8) (first y_8)) (or (second x_8) (second y_8))))

    func.func private @shl(%x : i32, %y : i32) -> i32 {
        %r = llvm.shl %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun shl ((x_9 (Pair (_ BitVec 32) Bool)) (y_9 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (let ((tmp (first y_9)))
    // CHECK-NEXT:    (pair (bvshl (first x_9) tmp) (or (bvugt tmp (_ bv32 32)) (or (second x_9) (second y_9))))))

    func.func private @ashr(%x : i32, %y : i32) -> i32 {
        %r = llvm.ashr %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun ashr ((x_10 (Pair (_ BitVec 32) Bool)) (y_10 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (let ((tmp (first y_10)))
    // CHECK-NEXT:    (pair (bvashr (first x_10) tmp) (or (bvugt tmp (_ bv32 32)) (or (second x_10) (second y_10))))))

    func.func private @lshr(%x : i32, %y : i32) -> i32 {
        %r = llvm.lshr %x, %y : i32
        func.return %r : i32
    }
    // CHECK-NEXT:  (define-fun lshr ((x_11 (Pair (_ BitVec 32) Bool)) (y_11 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
    // CHECK-NEXT:    (let ((tmp (first y_11)))
    // CHECK-NEXT:    (pair (bvlshr (first x_11) tmp) (or (bvugt tmp (_ bv32 32)) (or (second x_11) (second y_11))))))
}
