// RUN: xdsl-smt.py %s | xdsl-smt.py -t=smt | filecheck %s
// RUN: xdsl-smt.py %s -t=smt | z3 -in

"builtin.module"() ({
  %cst = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      (declare-const cst Bool)

  %fun = "smt.define_fun"() ({
  ^0(%x : !smt.bool):
    %y = "smt.not"(%x) : (!smt.bool) -> !smt.bool
    "smt.return"(%y) : (!smt.bool) -> ()
  }) : () -> ((!smt.bool) -> !smt.bool)
  // CHECK-NEXT: (define-fun fun ((x Bool)) Bool
  // CHECK-NEXT:   (not x))

  "smt.assert"(%cst) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert cst)

  "smt.check_sat"() : () -> ()
  // CHECK-NEXT: (check-sat)
}) : () -> ()
