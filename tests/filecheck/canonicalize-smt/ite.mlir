// RUN: xdsl-smt "%s" -p=canonicalize,dce -t=smt | filecheck "%s"

"builtin.module"() ({
  %true = "smt.constant_bool"() {"value" = #smt.bool_attr<true>} : () -> !smt.bool
  %false = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      (declare-const x Bool)
  %y = "smt.declare_const"() : () -> !smt.bool
  // CHECK-NEXT: (declare-const y Bool)

  // if true then x else y -> x
  %a = "smt.ite"(%true, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert x)

  // if false then x else y -> y
  %b = "smt.ite"(%false, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert y)

  // if c then y else y -> y
  %c = "smt.ite"(%x, %y, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert y)
}) : () -> ()
