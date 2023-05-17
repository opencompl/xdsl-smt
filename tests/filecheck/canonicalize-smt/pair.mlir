// RUN: xdsl-smt.py %s -p=canonicalize-smt -t=smt | filecheck %s

//CHECK:      (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))

"builtin.module"() ({
  %x = "smt.declare_const"() : () -> !smt.bool
  //CHECK-NEXT: (declare-const x Bool)
  %y = "smt.declare_const"() : () -> !smt.bool
  //CHECK-NEXT: (declare-const y Bool)

  %p = "smt.utils.pair"(%x, %y) : (!smt.bool, !smt.bool) -> !smt.utils.pair<!smt.bool, !smt.bool>
  %first = "smt.utils.first"(%p) : (!smt.utils.pair<!smt.bool, !smt.bool>) -> !smt.bool
  %second = "smt.utils.second"(%p) : (!smt.utils.pair<!smt.bool, !smt.bool>) -> !smt.bool

  // first (pair x y) -> x
  "smt.assert"(%first) : (!smt.bool) -> ()
  //CHECK-NEXT: (assert x)

  // second (pair x y) -> y
  "smt.assert"(%second) : (!smt.bool) -> ()
  //CHECK-NEXT: (assert y)
}) : () -> ()
