// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({
  %x = "smt.declare_const"() : () -> !smt.bv<8>
  %y = "smt.declare_const"() : () -> !smt.bv<8>
  %zero = smt.bv.constant #smt.bv<0> : !smt.bv<8>
  %five = smt.bv.constant #smt.bv<5> : !smt.bv<8>
  %c13 = smt.bv.constant #smt.bv<13> : !smt.bv<8>
  %m1 = smt.bv.constant #smt.bv<255> : !smt.bv<8>
  %m5 = smt.bv.constant #smt.bv<251> : !smt.bv<8>
  %m13 = smt.bv.constant #smt.bv<243> : !smt.bv<8>
  %int_min = smt.bv.constant #smt.bv<128> : !smt.bv<8>

  // Case 1: x sdiv 5  ==>  unchanged (rhs not zero, lhs not constant)
  %a = "smt.bv.sdiv"(%x, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %a_eq = "smt.eq"(%a, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%a_eq) : (!smt.bool) -> ()
  // CHECK: %a = "smt.bv.sdiv"(%x, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  // CHECK-NEXT: %a_eq = "smt.eq"(%a, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%a_eq) : (!smt.bool) -> ()

  // Case 2: x sdiv y  ==>  unchanged (rhs not a constant)
  %b = "smt.bv.sdiv"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %b_eq = "smt.eq"(%b, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%b_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %b = "smt.bv.sdiv"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  // CHECK-NEXT: %b_eq = "smt.eq"(%b, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%b_eq) : (!smt.bool) -> ()

  // -------- Constant folding (both operands constant) --------

  // Case 3: 13 sdiv 5  ==>  2  (folds)
  %c = "smt.bv.sdiv"(%c13, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %c_eq = "smt.eq"(%c, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%c_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %c = smt.bv.constant #smt.bv<2> : !smt.bv<8>
  // CHECK-NEXT: %c_eq = "smt.eq"(%c, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%c_eq) : (!smt.bool) -> ()

  // Case 4 (negative lhs): -13 sdiv 5  ==>  -2 (254 as 8-bit)  (folds)
  %d = "smt.bv.sdiv"(%m13, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %d_eq = "smt.eq"(%d, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%d_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %d = smt.bv.constant #smt.bv<254> : !smt.bv<8>
  // CHECK-NEXT: %d_eq = "smt.eq"(%d, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%d_eq) : (!smt.bool) -> ()

  // Case 5 (negative rhs): 13 sdiv -5  ==>  -2 (254 as 8-bit)  (folds)
  %e = "smt.bv.sdiv"(%c13, %m5) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %e_eq = "smt.eq"(%e, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%e_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %e = smt.bv.constant #smt.bv<254> : !smt.bv<8>
  // CHECK-NEXT: %e_eq = "smt.eq"(%e, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%e_eq) : (!smt.bool) -> ()

  // Case 6 (both negative): -13 sdiv -5  ==>  2  (folds)
  %f = "smt.bv.sdiv"(%m13, %m5) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %f_eq = "smt.eq"(%f, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%f_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %f = smt.bv.constant #smt.bv<2> : !smt.bv<8>
  // CHECK-NEXT: %f_eq = "smt.eq"(%f, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%f_eq) : (!smt.bool) -> ()

  // -------- Division by zero semantics (both operands constant) --------
  // If rhs is 0: result is 255 when lhs is positive, and 1 otherwise.

  // Case 7: 13 sdiv 0  ==>  255  (folds)
  %g = "smt.bv.sdiv"(%c13, %zero) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %g_eq = "smt.eq"(%g, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%g_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %g = smt.bv.constant #smt.bv<255> : !smt.bv<8>
  // CHECK-NEXT: %g_eq = "smt.eq"(%g, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%g_eq) : (!smt.bool) -> ()

  // Case 8: -13 sdiv 0  ==>  1  (folds)
  %h = "smt.bv.sdiv"(%m13, %zero) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %h_eq = "smt.eq"(%h, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%h_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %h = smt.bv.constant #smt.bv<1> : !smt.bv<8>
  // CHECK-NEXT: %h_eq = "smt.eq"(%h, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%h_eq) : (!smt.bool) -> ()

  // sdiv underflow semantics

  // Case 9: int_min sdiv -1  ==> int_min
  %i = "smt.bv.sdiv"(%int_min, %m1) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %i_eq = "smt.eq"(%i, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%i_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %i = smt.bv.constant #smt.bv<128> : !smt.bv<8>
  // CHECK-NEXT: %i_eq = "smt.eq"(%i, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%i_eq) : (!smt.bool) -> ()
}) : () -> ()
