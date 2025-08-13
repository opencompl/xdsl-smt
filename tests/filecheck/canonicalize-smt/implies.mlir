// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({

  %true = "smt.constant"() <{value = true}> : () -> !smt.bool
  %false = "smt.constant"() <{value = false}> : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK: %x = "smt.declare_const"() : () -> !smt.bool

  // (true => x) -> x
  %a = smt.implies  %true, %x
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // (false => x) -> true
  %b = smt.implies %false, %x
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: %b = "smt.constant"() <{value = true}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%b) : (!smt.bool) -> ()

  // (x => true) -> true
  %c = smt.implies %x, %true
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: %c = "smt.constant"() <{value = true}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%c) : (!smt.bool) -> ()

  // (x => false) -> not x
  %d = smt.implies %x, %false
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: %d = smt.not %x
  // CHECK-NEXT: "smt.assert"(%d) : (!smt.bool) -> ()

  // (x => x) -> true
  %e = smt.implies %x, %x
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: %e = "smt.constant"() <{value = true}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%e) : (!smt.bool) -> ()
}) : () -> ()
