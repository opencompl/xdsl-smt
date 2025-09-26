// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({
  %x = "smt.declare_const"() : () -> !smt.bv<8>
  %y = "smt.declare_const"() : () -> !smt.bv<8>
  %zero = smt.bv.constant #smt.bv<0> : !smt.bv<8>
  %five = smt.bv.constant #smt.bv<5> : !smt.bv<8>
  %c13 = smt.bv.constant #smt.bv<13> : !smt.bv<8>
  %cneg5 = smt.bv.constant #smt.bv<251> : !smt.bv<8>
  %cneg13 = smt.bv.constant #smt.bv<243> : !smt.bv<8>

  // Case 1: x srem 0  ==>  x     (folds to x)
  %a = "smt.bv.srem"(%x, %zero) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %a_eq = "smt.eq"(%a, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%a_eq) : (!smt.bool) -> ()
  // CHECK: %a_eq = "smt.eq"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%a_eq) : (!smt.bool) -> ()

  // Case 2: x srem 5  ==>  unchanged (rhs not zero, should NOT fold)
  %b = "smt.bv.srem"(%x, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %b_eq = "smt.eq"(%b, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%b_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %b = "smt.bv.srem"(%x, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  // CHECK-NEXT: %b_eq = "smt.eq"(%b, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%b_eq) : (!smt.bool) -> ()

  // Case 3: x srem y  ==>  unchanged (rhs not a constant, should NOT fold)
  %c = "smt.bv.srem"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %c_eq = "smt.eq"(%c, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%c_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %c = "smt.bv.srem"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  // CHECK-NEXT: %c_eq = "smt.eq"(%c, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%c_eq) : (!smt.bool) -> ()

  // Case 4: 0 srem 0  ==>  0   (folds to LHS operand, which is %zero)
  %d = "smt.bv.srem"(%zero, %zero) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %d_eq = "smt.eq"(%d, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%d_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %d_eq = "smt.eq"(%zero, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%d_eq) : (!smt.bool) -> ()

  // Case 5: const srem const  ==>  const result (13 srem 5 = 3)
  %e = "smt.bv.srem"(%c13, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %e_eq = "smt.eq"(%e, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%e_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %e = smt.bv.constant #smt.bv<3> : !smt.bv<8>
  // CHECK-NEXT: %e_eq = "smt.eq"(%e, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%e_eq) : (!smt.bool) -> ()

  // ---- Signed-specific cases with negative operands ----

  // Case 6: (-13) srem 5  ==>  2
  %f = "smt.bv.srem"(%cneg13, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %f_eq = "smt.eq"(%f, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%f_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %f = smt.bv.constant #smt.bv<2> : !smt.bv<8>
  // CHECK-NEXT: %f_eq = "smt.eq"(%f, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%f_eq) : (!smt.bool) -> ()

  // Case 7: 13 srem (-5)  ==>  -2
  %g = "smt.bv.srem"(%c13, %cneg5) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %g_eq = "smt.eq"(%g, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%g_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %g = smt.bv.constant #smt.bv<254> : !smt.bv<8>
  // CHECK-NEXT: %g_eq = "smt.eq"(%g, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%g_eq) : (!smt.bool) -> ()

  // Case 8: (-13) srem (-5)  ==>  -3
  %h = "smt.bv.srem"(%cneg13, %cneg5) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %h_eq = "smt.eq"(%h, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool // arbitrary eq to keep %h used
  "smt.assert"(%h_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %h = smt.bv.constant #smt.bv<253> : !smt.bv<8>
  // CHECK-NEXT: %h_eq = "smt.eq"(%h, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%h_eq) : (!smt.bool) -> ()

  // Case 9: x srem (-5)  ==>  unchanged (non-zero constant rhs, should NOT fold)
  %i = "smt.bv.srem"(%x, %cneg5) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %i_eq = "smt.eq"(%i, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%i_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %i = "smt.bv.srem"(%x, %cneg5) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  // CHECK-NEXT: %i_eq = "smt.eq"(%i, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%i_eq) : (!smt.bool) -> ()
}) : () -> ()

