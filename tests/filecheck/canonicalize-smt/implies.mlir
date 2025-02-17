// RUN: xdsl-smt "%s" -p=canonicalize,dce -t=smt | filecheck "%s"

"builtin.module"() ({

  %true = "smt.constant_bool"() {value = #smt.bool_attr<true>} : () -> !smt.bool
  %false = "smt.constant_bool"() {value = #smt.bool_attr<false>} : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      (declare-const x Bool)

  // (true => x) -> x
  %a = "smt.implies"(%true, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert x)

  // (false => x) -> true
  %b = "smt.implies"(%false, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert true)

  // (x => true) -> true
  %c = "smt.implies"(%x, %true) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert true)

  // (x => false) -> not x
  %d = "smt.implies"(%x, %false) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (not x))

  // (x => x) -> true
  %e = "smt.implies"(%x, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert true)
}) : () -> ()
