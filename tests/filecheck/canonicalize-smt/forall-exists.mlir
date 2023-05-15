// RUN: xdsl-smt.py %s -p=canonicalize-smt -t=smt | filecheck %s

"builtin.module"() ({
  // (forall x, true) -> true
  %b = "smt.forall"() ({
  ^0(%x : !smt.bool):
    %y = "smt.constant_bool"() {"value" = #smt.bool_attr<true>} : () -> !smt.bool
    "smt.yield"(%y) : (!smt.bool) -> ()
  }) : () -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK:      (assert true)

  // (forall x, false) -> false
  %c = "smt.forall"() ({
  ^1(%0 : !smt.bool):
    %y_1 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
    "smt.yield"(%y_1) : (!smt.bool) -> ()
  }) : () -> !smt.bool
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert false)

  // (exists x, true) -> true
  %d = "smt.exists"() ({
  ^2(%1 : !smt.bool):
    %y_2 = "smt.constant_bool"() {"value" = #smt.bool_attr<true>} : () -> !smt.bool
    "smt.yield"(%y_2) : (!smt.bool) -> ()
  }) : () -> !smt.bool
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert true)

  // (exists x, false) -> false
  %e = "smt.exists"() ({
  ^3(%2 : !smt.bool):
    %y_3 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
    "smt.yield"(%y_3) : (!smt.bool) -> ()
  }) : () -> !smt.bool
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert false)
}) : () -> ()

