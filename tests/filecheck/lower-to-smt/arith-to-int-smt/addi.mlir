// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i4):
  %y = "arith.constant"() {value = 3 : i4} : () -> i4
  %z = "arith.addi"(%x,%y): (i4,i4) -> i4
  "func.return"(%z) : (i4) -> ()
  }) {"sym_name" = "test", "function_type" = (i4) -> i4, "sym_visibility" = "private"} : () -> ()
}) : () -> ()
