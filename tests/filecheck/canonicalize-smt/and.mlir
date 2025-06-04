// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({
  %true = "smt.constant"() <{value = true}> : () -> !smt.bool
  %false = "smt.constant"() <{value = false}> : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK: %x = "smt.declare_const"() : () -> !smt.bool

  // true /\ x -> x
  %a = "smt.and"(%true, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // false /\ x -> false
  %b = "smt.and"(%false, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT:  %b = "smt.constant"() <{value = false}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%b) : (!smt.bool) -> ()

  // x /\ true -> x
  %c = "smt.and"(%x, %true) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // x /\ false -> false
  %d = "smt.and"(%x, %false) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: %d = "smt.constant"() <{value = false}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%d) : (!smt.bool) -> ()

  // x /\ x -> x
  %e = "smt.and"(%x, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()
}) : () -> ()
