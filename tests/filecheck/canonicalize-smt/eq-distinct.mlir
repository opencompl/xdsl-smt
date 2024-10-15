// RUN: xdsl-smt "%s" -p=canonicalize,dce -t=smt | filecheck "%s"

"builtin.module"() ({
  %true = "smt.constant_bool"() {"value" = !smt.bool_attr<true>} : () -> !smt.bool
  %false = "smt.constant_bool"() {"value" = !smt.bool_attr<false>} : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      (declare-const x Bool)

  // (true = x) -> x
  %a = "smt.eq"(%true, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert x)

  // (false = x) -> not x
  %b = "smt.eq"(%false, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (not x))

  // (x = true) -> x
  %c = "smt.eq"(%x, %true) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert x)

  // (x = false) -> not x
  %d = "smt.eq"(%x, %false) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (not x))

  // (x = x) -> true
  %e = "smt.eq"(%x, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert true)

  // (true != x) -> not x
  %f = "smt.distinct"(%true, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%f) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (not x))

  // (false != x) -> x
  %g = "smt.distinct"(%false, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%g) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert x)

  // (x != true) -> not x
  %h = "smt.distinct"(%x, %true) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%h) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (not x))

  // (x != false) -> x
  %i = "smt.distinct"(%x, %false) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%i) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert x)

  // (x != x) -> false
  %j = "smt.distinct"(%x, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%j) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert false)
}) : () -> ()
