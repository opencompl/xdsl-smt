// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({
  %true = "smt.constant"() <{value = true}> : () -> !smt.bool
  %false = "smt.constant"() <{value = false}> : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      %x = "smt.declare_const"() : () -> !smt.bool

  // (true = x) -> x
  %a = "smt.eq"(%true, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // (false = x) -> not x
  %b = "smt.eq"(%false, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: %b = smt.not %x
  // CHECK-NEXT: "smt.assert"(%b) : (!smt.bool) -> ()

  // (x = true) -> x
  %c = "smt.eq"(%x, %true) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // (x = false) -> not x
  %d = "smt.eq"(%x, %false) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: %d = smt.not %x
  // CHECK-NEXT: "smt.assert"(%d) : (!smt.bool) -> ()

  // (x = x) -> true
  %e = "smt.eq"(%x, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: %e = "smt.constant"() <{value = true}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%e) : (!smt.bool) -> ()

  // (true != x) -> not x
  %f = "smt.distinct"(%true, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%f) : (!smt.bool) -> ()
  // CHECK-NEXT: %f = smt.not %x
  // CHECK-NEXT: "smt.assert"(%f) : (!smt.bool) -> ()

  // (false != x) -> x
  %g = "smt.distinct"(%false, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%g) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // (x != true) -> not x
  %h = "smt.distinct"(%x, %true) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%h) : (!smt.bool) -> ()
  // CHECK-NEXT: %h = smt.not %x
  // CHECK-NEXT: "smt.assert"(%h) : (!smt.bool) -> ()

  // (x != false) -> x
  %i = "smt.distinct"(%x, %false) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%i) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // (x != x) -> false
  %j = "smt.distinct"(%x, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%j) : (!smt.bool) -> ()
  // CHECK-NEXT: %j = "smt.constant"() <{value = false}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%j) : (!smt.bool) -> ()
}) : () -> ()
