// RUN: xdsl-smt "%s" -p=pdl-to-smt,lower-effects,canonicalize,dce | filecheck "%s"

// or(x, or(x, y)) -> or(x, y)

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %2 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %3 = pdl.operation "arith.ori"(%1, %2 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
    %4 = "pdl.result"(%3) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    %5 = pdl.operation "arith.ori"(%1, %4 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
    pdl.rewrite %5 {
      pdl.replace %5 with (%4 : !pdl.value)
    }
  }) {benefit = 1 : i16} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %1 = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:    %2 = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:    %3 = "smt.utils.first"(%1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:    %4 = "smt.utils.second"(%1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:    %5 = "smt.utils.first"(%2) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:    %6 = "smt.utils.second"(%2) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:    %7 = "smt.or"(%4, %6) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %8 = "smt.bv.or"(%3, %5) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:    %9 = "smt.utils.first"(%1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:    %10 = "smt.utils.second"(%1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:    %11 = "smt.or"(%10, %7) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %12 = "smt.bv.or"(%9, %8) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:    %13 = "smt.not"(%11) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %14 = "smt.not"(%7) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %15 = "smt.eq"(%12, %8) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:    %16 = "smt.and"(%15, %14) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %17 = smt.implies %13, %16
// CHECK-NEXT:    %18 = "smt.not"(%0) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %19 = "smt.and"(%18, %17) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %20 = "smt.or"(%0, %19) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %21 = "smt.not"(%20) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    "smt.assert"(%21) : (!smt.bool) -> ()
// CHECK-NEXT:    "smt.check_sat"() : () -> ()
// CHECK-NEXT:  }
