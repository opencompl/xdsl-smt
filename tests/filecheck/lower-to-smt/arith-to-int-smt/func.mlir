// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects | filecheck %s
// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.addi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
