// RUN: xdsl-smt "%s" | xdsl-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt "%s" -t=smt | z3 -in

"builtin.module"() ({
  %true = "smt.constant_bool"() {value = #smt.bool_attr<true>} : () -> !smt.bool
  "smt.assert"(%true) : (!smt.bool) -> ()
  // CHECK:      (assert true)

  %x = "smt.declare_const"() : () -> !smt.bool
  %y = "smt.declare_const"() : () -> !smt.bool
  %z = "smt.declare_const"() : () -> !smt.bool
  // CHECK-NEXT: (declare-const $x Bool)
  // CHECK-NEXT: (declare-const $y Bool)
  // CHECK-NEXT: (declare-const $z Bool)

  %or = "smt.or"(%x, %y) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%or) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (or $x $y))

  %xor = "smt.xor"(%x, %y) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%xor) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (xor $x $y))

  "smt.check_sat"() : () -> ()
  // CHECK-NEXT: (check-sat)

  "smt.eval"(%or) : (!smt.bool) -> ()
  // CHECK-NEXT: (eval (or $x $y))

  "smt.eval"(%xor) : (!smt.bool) -> ()
  // CHECK-NEXT: (eval (xor $x $y))

  %and = "smt.and"(%x, %y) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%and) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (and $x $y))

  %distinct = "smt.distinct"(%x, %y) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%distinct) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (distinct $x $y))

  %eq = "smt.eq"(%x, %y) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%eq) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= $x $y))

  %implies = "smt.implies"(%x, %y) : (!smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%implies) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (=> $x $y))

  %ite = "smt.ite"(%x, %y, %z) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (ite $x $y $z))

  %not = "smt.not"(%x) : (!smt.bool) -> !smt.bool
  "smt.assert"(%not) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (not $x))

  %forall = "smt.forall"() ({
  ^0(%t : !smt.bool):
    "smt.yield"(%t) : (!smt.bool) -> ()
  }) : () -> !smt.bool
  "smt.assert"(%forall) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (forall (($t Bool)) $t))

  %exists = "smt.exists"() ({
  ^0(%t : !smt.bool):
    "smt.yield"(%t) : (!smt.bool) -> ()
  }) : () -> !smt.bool
  "smt.assert"(%exists) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (exists (({{.*}} Bool)) {{.*}}))

  %fun = "smt.define_fun"() ({
  ^0(%t : !smt.bool):
    "smt.return"(%t) : (!smt.bool) -> ()
  }) : () -> ((!smt.bool) -> !smt.bool)
  %res = "smt.call"(%fun, %true) : ((!smt.bool) -> !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%res) : (!smt.bool) -> ()
  // CHECK: (define-fun {{.*}} (({{.*}} Bool)) Bool
  // CHECK-NEXT: {{.*}})
  // CHECK-NEXT: (assert ({{.*}} true))
  // CHECK-NEXT: {{.*}})

  %false = "smt.constant_bool"() {value = #smt.bool_attr<false>} : () -> !smt.bool
  "smt.assert"(%false) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert false)

}) : () -> ()
