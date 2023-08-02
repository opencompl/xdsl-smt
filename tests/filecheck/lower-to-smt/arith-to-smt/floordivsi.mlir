// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.floordivsi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      (define-fun tmp ((x (_ BitVec 32)) (y (_ BitVec 32))) (_ BitVec 32)
// CHECK-NEXT:   (bvsdiv (bvsub x (ite (= ((_ extract 31 31) x) (_ bv1 1)) (bvsub y (_ bv1 32)) (_ bv0 32))) y))
