// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects | filecheck %s
// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : ui4):
  %y = "arith.constant"() {"value" = 3 : i64} : () -> ui4
  %z = "arith.cmpi" (%x, %z) {predicate = 8: i64} : (ui4,ui4) -> i1
  "func.return"(%z) : (i1) -> ()
  }) {"sym_name" = "test", "function_type" = (ui4) -> i1, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
