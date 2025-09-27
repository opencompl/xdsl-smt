// RUN: xdsl-smt "%s" -p=canonicalize,dce | filecheck "%s"

"builtin.module"() ({
  %x = "smt.declare_const"() : () -> !smt.bv<8>
  %y = "smt.declare_const"() : () -> !smt.bv<8>
  %zero = smt.bv.constant #smt.bv<0> : !smt.bv<8>
  %five = smt.bv.constant #smt.bv<5> : !smt.bv<8>
  %c13 = smt.bv.constant #smt.bv<13> : !smt.bv<8>

  // Case 1: x urem 0  ==>  x     (folds to x)
  %a = "smt.bv.urem"(%x, %zero) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %a_eq = "smt.eq"(%a, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%a_eq) : (!smt.bool) -> ()
  // CHECK: %a_eq = "smt.eq"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%a_eq) : (!smt.bool) -> ()

  // Case 2: x urem 5  ==>  unchanged (rhs not zero, should NOT fold)
  %b = "smt.bv.urem"(%x, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %b_eq = "smt.eq"(%b, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%b_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %b = "smt.bv.urem"(%x, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  // CHECK-NEXT: %b_eq = "smt.eq"(%b, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%b_eq) : (!smt.bool) -> ()

  // Case 3: x urem y  ==>  unchanged (rhs not a constant, should NOT fold)
  %c = "smt.bv.urem"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %c_eq = "smt.eq"(%c, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%c_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %c = "smt.bv.urem"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  // CHECK-NEXT: %c_eq = "smt.eq"(%c, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%c_eq) : (!smt.bool) -> ()

  // Case 4: 0 urem 0  ==>  0   (folds to LHS operand, which is %zero)
  %d = "smt.bv.urem"(%zero, %zero) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %d_eq = "smt.eq"(%d, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%d_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %d_eq = "smt.eq"(%zero, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%d_eq) : (!smt.bool) -> ()

  // Case 5: const urem const  ==>  const result (13 urem 5 = 3)
  %e = "smt.bv.urem"(%c13, %five) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
  %e_eq = "smt.eq"(%e, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  "smt.assert"(%e_eq) : (!smt.bool) -> ()
  // CHECK-NEXT: %e = smt.bv.constant #smt.bv<3> : !smt.bv<8>
  // CHECK-NEXT: %e_eq = "smt.eq"(%e, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
  // CHECK-NEXT: "smt.assert"(%e_eq) : (!smt.bool) -> ()
}) : () -> ()
