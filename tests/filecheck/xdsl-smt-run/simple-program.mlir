// RUN: xdsl-smt-run %s --args="#smt.bool_attr<true>" | FileCheck %s

%fun = "smt.define_fun"() ({
^0(%t : !smt.bool):
  %u = "smt.not"(%t) : (!smt.bool) -> !smt.bool
  "smt.return"(%u) : (!smt.bool) -> ()
}) : () -> ((!smt.bool) -> !smt.bool)

// CHECK: False
