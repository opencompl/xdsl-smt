// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({
  %true = smt.constant true
  %false = smt.constant false

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK: %x = "smt.declare_const"() : () -> !smt.bool

  // true /\ x -> x
  %a = smt.and %true, %x
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // false /\ x -> false
  %b = smt.and %false, %x
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT:  %b = smt.constant false
  // CHECK-NEXT: "smt.assert"(%b) : (!smt.bool) -> ()

  // x /\ true -> x
  %c = smt.and %x, %true
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // x /\ false -> false
  %d = smt.and %x, %false
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: %d = smt.constant false
  // CHECK-NEXT: "smt.assert"(%d) : (!smt.bool) -> ()

  // x /\ x -> x
  %e = smt.and %x, %x
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()
}) : () -> ()
