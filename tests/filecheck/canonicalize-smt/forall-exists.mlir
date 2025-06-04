// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({
  // (forall x, true) -> true
  %b = "smt.forall"() ({
  ^0(%x : !smt.bool):
    %y = "smt.constant"() <{value = true}> : () -> !smt.bool
    "smt.yield"(%y) : (!smt.bool) -> ()
  }) : () -> !smt.bool
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK:      %b = "smt.constant"() <{value = true}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%b) : (!smt.bool) -> ()

  // (forall x, false) -> false
  %c = "smt.forall"() ({
  ^1(%0 : !smt.bool):
    %y_1 = "smt.constant"() <{value = false}> : () -> !smt.bool
    "smt.yield"(%y_1) : (!smt.bool) -> ()
  }) : () -> !smt.bool
  "smt.assert"(%c) : (!smt.bool) -> ()
  // CHECK-NEXT: %c = "smt.constant"() <{value = false}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%c) : (!smt.bool) -> ()

  // (exists x, true) -> true
  %d = "smt.exists"() ({
  ^2(%1 : !smt.bool):
    %y_2 = "smt.constant"() <{value = true}> : () -> !smt.bool
    "smt.yield"(%y_2) : (!smt.bool) -> ()
  }) : () -> !smt.bool
  "smt.assert"(%d) : (!smt.bool) -> ()
  // CHECK-NEXT: %d = "smt.constant"() <{value = true}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%d) : (!smt.bool) -> ()

  // (exists x, false) -> false
  %e = "smt.exists"() ({
  ^3(%2 : !smt.bool):
    %y_3 = "smt.constant"() <{value = false}> : () -> !smt.bool
    "smt.yield"(%y_3) : (!smt.bool) -> ()
  }) : () -> !smt.bool
  "smt.assert"(%e) : (!smt.bool) -> ()
  // CHECK-NEXT: %e = "smt.constant"() <{value = false}> : () -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%e) : (!smt.bool) -> ()
}) : () -> ()
