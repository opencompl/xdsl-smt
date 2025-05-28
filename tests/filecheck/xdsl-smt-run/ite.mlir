// RUN: xdsl-smt-run %s --args="#smt.bool_attr<false>,#smt.bool_attr<false>,#smt.bool_attr<true>" | FileCheck %s

%fun = "smt.define_fun"() ({
^0(%cond : !smt.bool, %then : !smt.bool, %else : !smt.bool):
  %r = "smt.ite"(%cond, %then, %else) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.return"(%r) : (!smt.bool) -> ()
}) : () -> ((!smt.bool, !smt.bool, !smt.bool) -> !smt.bool)

// CHECK: True
