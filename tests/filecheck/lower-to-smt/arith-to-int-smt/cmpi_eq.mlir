// RUN: xdsl-smt %s -p=load-int-semantics,lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i4):
  %y = "arith.constant"() {value = 3 : i4} : () -> i4
  %z = "arith.cmpi" (%x, %y) {predicate = 0: i64} : (i4,i4) -> i1
  "func.return"() : () -> ()
  }) {"sym_name" = "test", "function_type" = (i4) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()
