// RUN: xdsl-smt "%s" -p=canonicalize,dce -t=smt | filecheck "%s"

"builtin.module"() ({

  %true = "smt.constant_bool"() {"value" = #smt.bool_attr<true>} : () -> !smt.bool
  %false = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      (declare-const x Bool)

  // true ^ x -> not x
  %a = "smt.xor"(%true, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (not x))

  // false ^ x -> x
  %b = "smt.xor"(%false, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert x)

  // x ^ true -> not x
  %c = "smt.xor"(%x, %true) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (not x))

  // x ^ false -> x
  %d = "smt.xor"(%x, %false) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert x)

  // x ^ x -> false
  %e = "smt.xor"(%x, %x) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert false)
}) : () -> ()
