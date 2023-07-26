// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"

"builtin.module"() ({
  "func.func"() ({
  ^0(%c : i1, %x : i32, %y : i32):
    %r = "arith.select"(%c, %x, %y) : (i1, i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i1, i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      (define-fun tmp ((c (_ BitVec 1)) (x (_ BitVec 32)) (y (_ BitVec 32))) (_ BitVec 32)
// CHECK-NEXT:   (ite (= c (_ bv1 1)) x y))
