// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects | filecheck %s
// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : ui4):
    %y = "arith.constant"() {"value" = 3 : i64} : () -> ui4
    %z = "arith.remui"(%x,%y): (ui4,ui4) -> ui4
   "func.return"(%z) : (ui4) -> ()
  }) {"sym_name" = "test", "function_type" = (ui4) -> ui4, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
