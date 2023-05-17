// RUN: xdsl-smt.py %s -p=arith-to-smt -t=smt | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32):
    "func.return"(%x) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()


// CHECK:      (define-fun tmp ((x (_ BitVec 32))) (_ BitVec 32)
// CHECK-NEXT:   x)
