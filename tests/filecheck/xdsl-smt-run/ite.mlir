// RUN: xdsl-smt-run %s --args="false,false,true" | FileCheck %s

func.func @main(%cond : !smt.bool, %then : !smt.bool, %else : !smt.bool) -> !smt.bool {
  %r = "smt.ite"(%cond, %then, %else) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  func.return %r : !smt.bool
}

// CHECK: True
