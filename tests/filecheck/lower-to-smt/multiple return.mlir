// RUN: xdsl-smt "%s" -p=lower-to-smt -t=smt | filecheck "%s"

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    "func.return"(%y, %x) : (i32, i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> (i32, i32), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      (define-fun tmp ((x (_ BitVec 32)) (y (_ BitVec 32))) (Pair (_ BitVec 32) (_ BitVec 32))
// CHECK-NEXT:   (pair y x))
