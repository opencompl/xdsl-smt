// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize-smt -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.ceildivsi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (define-fun test ((x (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (let ((tmp (first y)))
// CHECK-NEXT:    (let ((tmp_0 (first x)))
// CHECK-NEXT:    (let ((r (_ bv0 32)))
// CHECK-NEXT:    (let ((r_0 (_ bv1 32)))
// CHECK-NEXT:    (let ((r_1 (_ bv4294967295 32)))
// CHECK-NEXT:    (let ((r_2 (bvslt tmp_0 r)))
// CHECK-NEXT:    (pair (ite (or (and r_2 (bvslt tmp r)) (and (bvslt r tmp_0) (bvslt r tmp))) (bvadd (bvsdiv (bvsub tmp_0 (ite r_2 r_1 r_0)) tmp) r_0) (bvsub r (bvsdiv (bvsub r tmp_0) tmp))) (or (or (and (= tmp_0 (_ bv2147483648 32)) (= tmp r_1)) (= r tmp)) (or (second x) (second y)))))))))))
