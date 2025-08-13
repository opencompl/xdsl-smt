// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({
  %true = "smt.constant"() <{value = true}> : () -> !smt.bool
  %false = "smt.constant"() <{value = false}> : () -> !smt.bool

  %x = "smt.declare_const"() : () -> !smt.bool
  %y = "smt.declare_const"() : () -> !smt.bool
  %c = "smt.declare_const"() : () -> !smt.bool
  %c2 = "smt.declare_const"() : () -> !smt.bool
  // CHECK:      %x = "smt.declare_const"() : () -> !smt.bool
  // CHECK-NEXT: %y = "smt.declare_const"() : () -> !smt.bool
  // CHECK-NEXT: %c = "smt.declare_const"() : () -> !smt.bool
  // CHECK-NEXT: %c2 = "smt.declare_const"() : () -> !smt.bool

  // if true then x else y -> x
  %ite1 = "smt.ite"(%true, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite1) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%x) : (!smt.bool) -> ()

  // if false then x else y -> y
  %ite2 = "smt.ite"(%false, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite2) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%y) : (!smt.bool) -> ()

  // if c then y else y -> y
  %ite3 = "smt.ite"(%x, %y, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite3) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%y) : (!smt.bool) -> ()

  // ((x if c else y) if c' else y) -> x if c && c' else y
  %t4 = "smt.ite"(%c, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  %ite4 = "smt.ite"(%c2, %t4, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite4) : (!smt.bool) -> ()
  // CHECK-NEXT: %ite4 = smt.and %c2, %c
  // CHECK-NEXT: %ite4_1 = "smt.ite"(%ite4, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%ite4_1) : (!smt.bool) -> ()

  // ((x if c else y) if c' else x) -> y if c' && !c else x
  %t5 = "smt.ite"(%c, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  %ite5 = "smt.ite"(%c2, %t5, %x) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite5) : (!smt.bool) -> ()
  // CHECK-NEXT: %ite5 = smt.not %c
  // CHECK-NEXT: %ite5_1 = smt.and %c2, %ite5
  // CHECK-NEXT: %ite5_2 = "smt.ite"(%ite5_1, %y, %x) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%ite5_2) : (!smt.bool) -> ()

  // (x if c else (x if c' else y)) -> x if c || c' else y
  %t6 = "smt.ite"(%c2, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  %ite6 = "smt.ite"(%c, %x, %t6) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite6) : (!smt.bool) -> ()
  // CHECK-NEXT: %ite6 = smt.or %c, %c2
  // CHECK-NEXT: %ite6_1 = "smt.ite"(%ite6, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%ite6_1) : (!smt.bool) -> ()

  // (x if c else (y if c' else x)) -> x if c || !c' else y
  %t7 = "smt.ite"(%c2, %y, %x) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  %ite7 = "smt.ite"(%c, %x, %t7) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  "smt.assert"(%ite7) : (!smt.bool) -> ()
  // CHECK-NEXT: %ite7 = smt.not %c2
  // CHECK-NEXT: %ite7_1 = smt.or %c, %ite7
  // CHECK-NEXT: %ite7_2 = "smt.ite"(%ite7_1, %x, %y) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%ite7_2) : (!smt.bool) -> ()
}) : () -> ()
