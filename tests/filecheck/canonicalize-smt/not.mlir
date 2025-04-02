// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({

  %true = "smt.constant_bool"() {value = #smt.bool_attr<true>} : () -> !smt.bool
  %false = "smt.constant_bool"() {value = #smt.bool_attr<false>} : () -> !smt.bool

  // not true -> false
  %a = "smt.not"(%true) : (!smt.bool) -> !smt.bool
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK:      %a = "smt.constant_bool"() {value = #smt.bool_attr<false>} : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%a) : (!smt.bool) -> ()

  // not false -> true
  %b = "smt.not"(%false) : (!smt.bool) -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: %b = "smt.constant_bool"() {value = #smt.bool_attr<true>} : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%b) : (!smt.bool) -> ()
}) : () -> ()
