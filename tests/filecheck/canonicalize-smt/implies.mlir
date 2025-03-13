// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({

  %true = "smt.constant_bool"() {value = #smt.bool_attr<true>} : () -> !smt.bool
  %false = "smt.constant_bool"() {value = #smt.bool_attr<false>} : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK: %x = "smt.declare_const"() : () -> !smt.bool

  // (true => x) -> x
  %a = "smt.implies"(%true, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // (false => x) -> true
  %b = "smt.implies"(%false, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: %b = "smt.constant_bool"() {value = #smt.bool_attr<true>} : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%b) : (!smt.bool) -> ()

  // (x => true) -> true
  %c = "smt.implies"(%x, %true) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: %c = "smt.constant_bool"() {value = #smt.bool_attr<true>} : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%c) : (!smt.bool) -> ()

  // (x => false) -> not x
  %d = "smt.implies"(%x, %false) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: %d = "smt.not"(%x) : (!smt.bool) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%d) : (!smt.bool) -> ()

  // (x => x) -> true
  %e = "smt.implies"(%x, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: %e = "smt.constant_bool"() {value = #smt.bool_attr<true>} : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%e) : (!smt.bool) -> ()
}) : () -> ()
