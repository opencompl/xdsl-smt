// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = arith.cmpi slt, %x, %y : i32
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i1, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      (define-fun tmp ((x (_ BitVec 32)) (y (_ BitVec 32))) (_ BitVec 1)
// CHECK-NEXT:   (ite (bvslt x y) (_ bv1 1) (_ bv0 1)))
