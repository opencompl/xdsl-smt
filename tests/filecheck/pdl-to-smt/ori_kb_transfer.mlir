// RUN: xdsl-smt.py %s -t mlir -p pdl-to-smt | filecheck %s

"builtin.module"() ({
  "pdl.pattern"() ({
    // Get an i32 type
    %type = "pdl.type"() {"constantType" = i32} : () -> !pdl.type

    // Get two i32 values that have a known bits pattern
    %lhs, %lhs_zeros, %lhs_ones = "pdl.kb.operand"(%type) : (!pdl.type) -> (!pdl.value, i32, i32)
    %rhs, %rhs_zeros, %rhs_ones = "pdl.kb.operand"(%type) : (!pdl.type) -> (!pdl.value, i32, i32)

    // Get an add operation that takes both operands
    %op = "pdl.operation"(%lhs, %rhs, %type) {opName = "arith.ori", attributeValueNames = [], operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation

    "pdl.rewrite"(%op) ({
        // Compute the known bits of the result
        %res_zeros = "arith.andi"(%lhs_zeros, %rhs_zeros) : (i32, i32) -> i32
        %res_ones = "arith.ori"(%lhs_ones, %rhs_ones) : (i32, i32) -> i32

        // Attach it to the operation
        "pdl.kb.attach"(%op, %res_zeros, %res_ones) : (!pdl.operation, i32, i32) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "addi_analysis"} : () -> ()
}) : () -> ()


// CHECK: "builtin.module"() ({
// CHECK-NEXT:   %lhs = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %lhs_zeros = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %lhs_ones = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %lhs_1 = "smt.bv.and"(%lhs, %lhs_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %lhs_2 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %lhs_3 = "smt.eq"(%lhs_1, %lhs_2) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %lhs_4 = "smt.bv.and"(%lhs, %lhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %lhs_5 = "smt.eq"(%lhs_4, %lhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %lhs_6 = "smt.and"(%lhs_3, %lhs_5) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %rhs = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %rhs_zeros = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %rhs_ones = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %rhs_1 = "smt.bv.and"(%rhs, %rhs_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %rhs_2 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %rhs_3 = "smt.eq"(%rhs_1, %rhs_2) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %rhs_4 = "smt.bv.and"(%rhs, %rhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %rhs_5 = "smt.eq"(%rhs_4, %rhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %rhs_6 = "smt.and"(%rhs_3, %rhs_5) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %0 = "smt.bv.or"(%lhs, %rhs) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %res_zeros = "smt.bv.and"(%lhs_zeros, %rhs_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %res_ones = "smt.bv.or"(%lhs_ones, %rhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %1 = "smt.bv.and"(%0, %res_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %2 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %3 = "smt.eq"(%1, %2) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %4 = "smt.bv.and"(%0, %res_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %5 = "smt.eq"(%4, %res_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %6 = "smt.and"(%3, %5) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %7 = "smt.not"(%6) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:   %8 = "smt.and"(%lhs_6, %rhs_6) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %9 = "smt.and"(%8, %7) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   "smt.assert"(%9) : (!smt.bool) -> ()
// CHECK-NEXT:   "smt.check_sat"() : () -> ()
// CHECK-NEXT: }) : () -> ()
