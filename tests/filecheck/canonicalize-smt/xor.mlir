// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({
  %true = "smt.constant"() <{value = true}> : () -> !smt.bool
  %false = "smt.constant"() <{value = false}> : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      %x = "smt.declare_const"() : () -> !smt.bool

  // true ^ x -> not x
  %a = "smt.xor"(%true, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK-NEXT: %a = smt.not %x
  // CHECK-NEXT: "smt.assert"(%a) : (!smt.bool) -> ()

  // false ^ x -> x
  %b = "smt.xor"(%false, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // x ^ true -> not x
  %c = "smt.xor"(%x, %true) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: %c = smt.not %x
  // CHECK-NEXT: "smt.assert"(%c) : (!smt.bool) -> ()

  // x ^ false -> x
  %d = "smt.xor"(%x, %false) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // x ^ x -> false
  %e = "smt.xor"(%x, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: %e = "smt.constant"() <{value = false}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%e) : (!smt.bool) -> ()
}) : () -> ()
