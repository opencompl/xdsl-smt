// RUN: xdsl-smt "%s" -p=lower-to-smt,lower-effects -t=smt | filecheck "%s"

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    "func.return"(%y, %x) : (i32, i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> (i32, i32), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      (define-fun test ((x (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool)) (tmp Bool)) (Pair (Pair (_ BitVec 32) Bool) (Pair (Pair (_ BitVec 32) Bool) Bool))
// CHECK-NEXT:   (pair y (pair x tmp)))
