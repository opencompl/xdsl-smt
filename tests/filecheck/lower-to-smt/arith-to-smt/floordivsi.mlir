// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize-smt -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.floordivsi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (define-fun test ((x (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (let ((tmp (first y)))
// CHECK-NEXT:    (let ((tmp_0 (first x)))
// CHECK-NEXT:    (let ((r (bvsdiv tmp_0 tmp)))
// CHECK-NEXT:    (let ((r_0 (_ bv4294967295 32)))
// CHECK-NEXT:    (let ((r_1 (_ bv0 32)))
// CHECK-NEXT:    (pair (ite (and (xor (bvslt tmp_0 r_1) (bvslt tmp r_1)) (distinct (bvsrem tmp_0 tmp) r_1)) (bvadd r r_0) r) (or (or (and (= tmp_0 (_ bv2147483648 32)) (= tmp r_0)) (= r_1 tmp)) (or (second x) (second y))))))))))
