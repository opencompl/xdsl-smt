// RUN: xdsl-smt "%s" -p=pdl-to-smt,lower-effects,canonicalize,dce | filecheck "%s"

builtin.module {
    pdl.pattern @add_constant_fold : benefit(0) {
        %type = pdl.type : i32

        %c0 = pdl.attribute : %type
        %c1 = pdl.attribute : %type

        %constant0 = pdl.operation "arith.constant" {"value" = %c0} -> (%type : !pdl.type)
        %constant1 = pdl.operation "arith.constant" {"value" = %c1} -> (%type : !pdl.type)

        %lhs = pdl.result 0 of %constant0
        %rhs = pdl.result 0 of %constant1

        %no_overflow = pdl.attribute = #arith.overflow<none>

        %add = pdl.operation "arith.addi" (%lhs, %rhs : !pdl.value, !pdl.value) {"overflowFlags" = %no_overflow} -> (%type : !pdl.type)

        pdl.rewrite %add {
            %res = pdl.apply_native_rewrite "addi"(%c0, %c1 : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %res_constant = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            pdl.replace %add with %res_constant
        }
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %c0 = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:    %c1 = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:    %1 = "smt.bv.add"(%c0, %c1) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %res = "smt.bv.add"(%c0, %c1) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %2 = "smt.eq"(%1, %res) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:    %3 = "smt.not"(%0) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %4 = "smt.and"(%3, %2) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %5 = "smt.or"(%0, %4) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %6 = "smt.not"(%5) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    "smt.assert"(%6) : (!smt.bool) -> ()
// CHECK-NEXT:    "smt.check_sat"() : () -> ()
// CHECK-NEXT:  }
