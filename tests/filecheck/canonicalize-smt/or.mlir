// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({
  %true = smt.constant true
  %false = smt.constant false

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      %x = "smt.declare_const"() : () -> !smt.bool

  // true \/ x -> true
  %a = smt.or %true, %x
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK-NEXT: %a = smt.constant true
  // CHECK-NEXT: "smt.assert"(%a) : (!smt.bool) -> ()

  // false \/ x -> x
  %b = smt.or %false, %x
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // x \/ true -> true
  %c = smt.or %x, %true
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: %c = smt.constant true
  // CHECK-NEXT: "smt.assert"(%c) : (!smt.bool) -> ()

  // x \/ false -> x
  %d = smt.or %x, %false
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // x \/ x -> x
  %e = smt.or %x, %x
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()
}) : () -> ()
