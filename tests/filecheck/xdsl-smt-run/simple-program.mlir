// RUN: xdsl-smt-run %s --args="true" | FileCheck %s

func.func @main(%t : !smt.bool) -> !smt.bool {
  %u = smt.not %t
  func.return %u : !smt.bool
}

// CHECK: False
