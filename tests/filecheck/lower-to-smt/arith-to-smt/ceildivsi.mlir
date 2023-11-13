// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize-smt -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.ceildivsi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT: (define-fun test ((x (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:   (let ((tmp (first y)))
// CHECK-NEXT:   (let ((r (_ bv1 32)))
// CHECK-NEXT:   (let ((tmp_0 (first x)))
// CHECK-NEXT:   (pair (ite (= tmp_0 (_ bv2147483648 32)) (bvsdiv (bvadd tmp_0 (bvsub tmp r)) tmp) (bvadd r (bvsdiv (bvsub tmp_0 r) tmp))) (or (second x) (second y)))))))
