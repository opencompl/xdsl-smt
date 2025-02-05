// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : ui4):
  %y = "arith.constant"() {value = 3 : ui4} : () -> ui4
  %z = "arith.muli"(%x,%y): (ui4,ui4) -> ui4
  "func.return"(%z) : (ui4) -> ()
  }) {"sym_name" = "test", "function_type" = (ui4) -> ui4, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
