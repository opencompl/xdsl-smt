// RUN: xdsl-smt "%s" -p=pdl-to-smt,lower-effects,canonicalize,dce | filecheck "%s"

builtin.module {
    // x * -1 -> 0 - x
    pdl.pattern @mul_minus_one : benefit(0) {
        %type = pdl.type : i32

        %x = pdl.operand : %type
        %c0_attr = pdl.attribute : %type

        pdl.apply_native_constraint "is_minus_one"(%c0_attr : !pdl.attribute)

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)

        %c0 = pdl.result 0 of %c0_op

        %add = pdl.operation "arith.muli" (%x, %c0 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add {
            %zero_attr = pdl.attribute = 0 : i32
            %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%type : !pdl.type)
            %zero = pdl.result 0 of %zero_op
            %res = pdl.operation "arith.subi"(%zero, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %add with %res
        }
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %x = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
// CHECK-NEXT:    %c0_attr = "smt.declare_const"() : () -> !smt.bv<32>
// CHECK-NEXT:    %1 = "smt.bv.constant"() {value = #smt.bv.bv_val<4294967295: 32>} : () -> !smt.bv<32>
// CHECK-NEXT:    %2 = "smt.eq"(%c0_attr, %1) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:    %3 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:    %4 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:    %5 = "smt.bv.mul"(%3, %c0_attr) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:    %zero_attr = "smt.bv.constant"() {value = #smt.bv.bv_val<0: 32>} : () -> !smt.bv<32>
// CHECK-NEXT:    %6 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
// CHECK-NEXT:    %7 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:    %8 = "smt.bv.sub"(%zero_attr, %6) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:    %9 = "smt.not"(%4) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %10 = "smt.not"(%7) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %11 = "smt.eq"(%5, %8) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:    %12 = smt.and %11, %10
// CHECK-NEXT:    %13 = smt.implies %9, %12
// CHECK-NEXT:    %14 = "smt.not"(%0) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %15 = smt.and %14, %13
// CHECK-NEXT:    %16 = smt.or %0, %15
// CHECK-NEXT:    %17 = "smt.not"(%16) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %18 = smt.and %17, %2
// CHECK-NEXT:    "smt.assert"(%18) : (!smt.bool) -> ()
// CHECK-NEXT:    "smt.check_sat"() : () -> ()
// CHECK-NEXT:  }
