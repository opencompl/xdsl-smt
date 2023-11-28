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
// CHECK-NEXT:    (let ((tmp_1 (_ bv0 32)))
// CHECK-NEXT:    (let ((tmp_2 (_ bv1 32)))
// CHECK-NEXT:    (let ((tmp_3 (_ bv4294967295 32)))
// CHECK-NEXT:    (let ((tmp_4 (bvslt tmp_0 tmp_1)))
// CHECK-NEXT:    (pair (ite (or (and tmp_4 (bvslt tmp tmp_1)) (and (bvslt tmp_1 tmp_0) (bvslt tmp_1 tmp))) (bvadd (bvsdiv (bvsub tmp_0 (ite tmp_4 tmp_3 tmp_2)) tmp) tmp_2) (bvsub tmp_1 (bvsdiv (bvsub tmp_1 tmp_0) tmp))) (or (or (and (= tmp_0 (_ bv2147483648 32)) (= tmp tmp_3)) (= tmp_1 tmp)) (or (second x) (second y)))))))))))
