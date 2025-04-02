// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({
  %x = "smt.declare_const"() : () -> !smt.bool
  %y = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK-NEXT: %y = "smt.declare_const"() : () -> !smt.bool

  %p = "smt.utils.pair"(%x, %y) : (!smt.bool, !smt.bool) -> !smt.utils.pair<!smt.bool, !smt.bool>
  %first = "smt.utils.first"(%p) : (!smt.utils.pair<!smt.bool, !smt.bool>) -> !smt.bool
  %second = "smt.utils.second"(%p) : (!smt.utils.pair<!smt.bool, !smt.bool>) -> !smt.bool

  // first (pair x y) -> x
  "smt.assert"(%first) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // second (pair x y) -> y
  "smt.assert"(%second) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%y) : (!smt.bool) -> ()
}) : () -> ()
