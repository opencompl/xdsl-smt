// RUN: xdsl-smt.py %s -p=canonicalize-smt -t=smt | filecheck %s

"builtin.module"() ({
  %true = "smt.constant_bool"() {"value" = #smt.bool_attr<true>} : () -> !smt.bool
  %false = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool

  // true /\ x -> x
  %a = "smt.and"(%true, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK:      (declare-const x Bool)

  // false /\ x -> false
  %b = "smt.and"(%false, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert x)

  // x /\ true -> x
  %c = "smt.and"(%x, %true) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert false)

  // x /\ false -> false
  %d = "smt.and"(%x, %false) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert x)

  // x /\ x -> x
  %e = "smt.and"(%x, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert false)
}) : () -> ()
