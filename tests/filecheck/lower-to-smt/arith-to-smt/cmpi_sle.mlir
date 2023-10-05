// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize-smt -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.cmpi"(%x, %y) {"predicate" = 3 : i64} : (i32, i32) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i1, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (define-fun test ((x (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 1) Bool)
// CHECK-NEXT:    (pair (ite (bvsle (first x) (first y)) (_ bv1 1) (_ bv0 1)) (or (second x) (second y))))
