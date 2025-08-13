// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({

  %true = smt.constant true
  %false = smt.constant false

  // not true -> false
  %a = smt.not %true
  "smt.assert"(%a) : (!smt.bool) -> ()
  // CHECK:      %a = smt.constant false
  // CHECK-NEXT: "smt.assert"(%a) : (!smt.bool) -> ()

  // not false -> true
  %b = smt.not %false
  "smt.assert"(%b) : (!smt.bool) -> ()
  // CHECK-NEXT: %b = smt.constant true
  // CHECK-NEXT: "smt.assert"(%b) : (!smt.bool) -> ()
}) : () -> ()
