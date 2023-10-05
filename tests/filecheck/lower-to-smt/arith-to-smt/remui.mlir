// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize-smt -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.remui"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (define-fun test ((x (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (pair (bvurem (first x) (first y)) (or (second x) (second y))))
