// RUN: xdsl-smt "%s" -p=lower-pairs | filecheck "%s"

// Check that pair function arguments are split into multiple arguments in a forall loop

"builtin.module"() ({
  %0 = "smt.forall"() ({
^0(%1 : !smt.utils.pair<!smt.bv<4>, !smt.bool>, %2 : !smt.utils.pair<!smt.bv<4>, !smt.bool>):
  %3 = "smt.utils.first"(%1) : (!smt.utils.pair<!smt.bv<4>, !smt.bool>) -> !smt.bv<4>
  %4 = "smt.utils.first"(%2) : (!smt.utils.pair<!smt.bv<4>, !smt.bool>) -> !smt.bv<4>
  %5 = "smt.utils.second"(%1) : (!smt.utils.pair<!smt.bv<4>, !smt.bool>) -> !smt.bool
  %6 = "smt.utils.second"(%2) : (!smt.utils.pair<!smt.bv<4>, !smt.bool>) -> !smt.bool
  %7 = "smt.eq"(%3, %4) : (!smt.bv<4>, !smt.bv<4>) -> !smt.bool
  %8 = smt.and %5, %6
  %9 = smt.implies %7, %8
  "smt.yield"(%9) : (!smt.bool) -> ()
}) : () -> !smt.bool
  "smt.assert"(%0) : (!smt.bool) -> ()
}) : () -> ()

// CHECK:      %0 = "smt.forall"() ({
// CHECK-NEXT: ^0(%1 : !smt.bv<4>, %2 : !smt.bool, %3 : !smt.bv<4>, %4 : !smt.bool):
// CHECK-NEXT:   %5 = "smt.eq"(%1, %3) : (!smt.bv<4>, !smt.bv<4>) -> !smt.bool
// CHECK-NEXT:   %6 = smt.and %2, %4
// CHECK-NEXT:   %7 = smt.implies %5, %6
// CHECK-NEXT:   "smt.yield"(%7) : (!smt.bool) -> ()
// CHECK-NEXT: }) : () -> !smt.bool
// CHECK-NEXT: "smt.assert"(%0) : (!smt.bool) -> ()
