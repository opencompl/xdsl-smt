// RUN: xdsl-smt "%s" -p=canonicalize,dce -t=smt | filecheck "%s"

"builtin.module"() ({
  %true = "smt.constant_bool"() {"value" = #smt.bool_attr<true>} : () -> !smt.bool
  %false = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      (declare-const x Bool)
  %y = "smt.declare_const"() : () -> !smt.bool
  // CHECK-NEXT: (declare-const y Bool)
  %c = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      (declare-const c Bool)
  %c2 = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      (declare-const c2 Bool)


  // if true then x else y -> x
  %ite1 = "smt.ite"(%true, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite1) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert x)

  // if false then x else y -> y
  %ite2 = "smt.ite"(%false, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite2) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert y)

  // if c then y else y -> y
  %ite3 = "smt.ite"(%x, %y, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite3) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert y)

  // ((x if c else y) if c' else y) -> x if c && c' else y
  %t4 = "smt.ite"(%c, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  %ite4 = "smt.ite"(%c2, %t4, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite4) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (ite (and c2 c) x y))

  // ((x if c else y) if c' else x) -> y if c' && !c else x
  %t5 = "smt.ite"(%c, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  %ite5 = "smt.ite"(%c2, %t5, %x) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite5) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (ite (and c2 (not c)) x y))

  // (x if c else (x if c' else y)) -> x if c || c' else y
  %t6 = "smt.ite"(%c2, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  %ite6 = "smt.ite"(%c, %x, %t6) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite6) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (ite (or c c2) x y))

  // (x if c else (y if c' else x)) -> x if c || !c' else y
  %t7 = "smt.ite"(%c2, %y, %x) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  %ite7 = "smt.ite"(%c, %x, %t7) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite7) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (ite (or c (not c2)) x y))
}) : () -> ()
