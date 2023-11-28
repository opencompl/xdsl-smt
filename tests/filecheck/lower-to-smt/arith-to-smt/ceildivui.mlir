// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize-smt -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.ceildivui"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (define-fun test ((x (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (let ((tmp (_ bv1 32)))
// CHECK-NEXT:    (let ((tmp_0 (first y)))
// CHECK-NEXT:    (let ((tmp_1 (first x)))
// CHECK-NEXT:    (let ((tmp_2 (_ bv0 32)))
// CHECK-NEXT:    (pair (ite (= tmp_2 tmp_1) tmp_2 (bvadd (bvudiv (bvsub tmp_1 tmp) tmp_0) tmp)) (or (= tmp_2 tmp_0) (or (second x) (second y)))))))))
