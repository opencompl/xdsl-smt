// RUN: xdsl-smt "%s" -p=pdl-to-smt,lower-effects,canonicalize,dce | filecheck "%s"

// or(x, y) -> or(y, x)

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %2 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %3 = pdl.operation "arith.ori"(%1, %2 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
    pdl.rewrite %3 {
      %5 = pdl.operation "arith.ori"(%2, %1 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
      pdl.replace %3 with %5
    }
  }) {benefit = 1 : i16} : () -> ()
}) : () -> ()


// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %1 = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:    %2 = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:    %3 = "smt.utils.first"(%1) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %4 = "smt.utils.second"(%1) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:    %5 = "smt.utils.first"(%2) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %6 = "smt.utils.second"(%2) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:    %7 = "smt.or"(%4, %6) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %8 = "smt.bv.or"(%3, %5) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %9 = "smt.utils.first"(%2) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %10 = "smt.utils.second"(%2) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:    %11 = "smt.utils.first"(%1) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %12 = "smt.utils.second"(%1) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:    %13 = "smt.or"(%10, %12) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %14 = "smt.bv.or"(%9, %11) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %15 = "smt.not"(%7) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %16 = "smt.not"(%13) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %17 = "smt.eq"(%8, %14) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:    %18 = "smt.and"(%17, %16) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %19 = "smt.implies"(%15, %18) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %20 = "smt.not"(%0) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %21 = "smt.and"(%20, %19) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %22 = "smt.or"(%0, %21) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %23 = "smt.not"(%22) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    "smt.assert"(%23) : (!smt.bool) -> ()
// CHECK-NEXT:    "smt.check_sat"() : () -> ()
// CHECK-NEXT:  }