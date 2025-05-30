// RUN: xdsl-smt-run %s --args="#smt.bool_attr<false>,#smt.bool_attr<false>,#smt.bool_attr<true>" | FileCheck %s

func.func @main(%cond : !smt.bool, %then : !smt.bool, %else : !smt.bool) -> !smt.bool {
  %r = "smt.ite"(%cond, %then, %else) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  func.return %r : !smt.bool
}

// CHECK: True
