// RUN: xdsl-smt "%s" -t mlir -p pdl-to-smt | filecheck "%s"


builtin.module {
  pdl.pattern @addi_analysis : benefit(0) {
    // Get an i16 type
    %type = pdl.type : i16

    // Get the two operation operands
    %lhs = pdl.operand : %type
    %rhs = pdl.operand : %type

    // Get the operands analysis values
    %lhs_zeros, %lhs_ones = "pdl.dataflow.get"(%lhs) {"domain_name" = "kb"} : (!pdl.value) -> (!transfer.integer, !transfer.integer)
    %rhs_zeros, %rhs_ones = "pdl.dataflow.get"(%rhs) {"domain_name" = "kb"} : (!pdl.value) -> (!transfer.integer, !transfer.integer)

    // Get an add operation that takes both operands
    %op = pdl.operation "arith.ori"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%type : !pdl.type)

    "pdl.dataflow.rewrite"(%op) ({
        // Compute the known bits of the result
        %res_zeros = "transfer.and"(%lhs_zeros, %rhs_zeros) : (!transfer.integer, !transfer.integer) -> !transfer.integer
        %res_ones = "transfer.or"(%lhs_ones, %rhs_ones) : (!transfer.integer, !transfer.integer) -> !transfer.integer

        // Get the value of the operation result
        %res = pdl.result 0 of %op

        // Attach it to the operation
        "pdl.dataflow.attach"(%res, %res_zeros, %res_ones) {"domain_name" = "kb"} : (!pdl.value, !transfer.integer, !transfer.integer) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }
}


// CHECK:      builtin.module {
// CHECK-NEXT:   %lhs = "smt.declare_const"() : () -> !smt.bv.bv<16>
// CHECK-NEXT:   %rhs = "smt.declare_const"() : () -> !smt.bv.bv<16>
// CHECK-NEXT:   %lhs_zeros = "smt.declare_const"() : () -> !smt.bv.bv<16>
// CHECK-NEXT:   %lhs_ones = "smt.declare_const"() : () -> !smt.bv.bv<16>
// CHECK-NEXT:   %lhs_zeros_1 = "smt.bv.and"(%lhs, %lhs_zeros) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bv.bv<16>
// CHECK-NEXT:   %lhs_zeros_2 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 16>} : () -> !smt.bv.bv<16>
// CHECK-NEXT:   %lhs_zeros_3 = "smt.eq"(%lhs_zeros_1, %lhs_zeros_2) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bool
// CHECK-NEXT:   %lhs_zeros_4 = "smt.bv.and"(%lhs, %lhs_ones) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bv.bv<16>
// CHECK-NEXT:   %lhs_zeros_5 = "smt.eq"(%lhs_zeros_4, %lhs_ones) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bool
// CHECK-NEXT:   %lhs_zeros_6 = "smt.and"(%lhs_zeros_3, %lhs_zeros_5) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %rhs_zeros = "smt.declare_const"() : () -> !smt.bv.bv<16>
// CHECK-NEXT:   %rhs_ones = "smt.declare_const"() : () -> !smt.bv.bv<16>
// CHECK-NEXT:   %rhs_zeros_1 = "smt.bv.and"(%rhs, %rhs_zeros) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bv.bv<16>
// CHECK-NEXT:   %rhs_zeros_2 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 16>} : () -> !smt.bv.bv<16>
// CHECK-NEXT:   %rhs_zeros_3 = "smt.eq"(%rhs_zeros_1, %rhs_zeros_2) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bool
// CHECK-NEXT:   %rhs_zeros_4 = "smt.bv.and"(%rhs, %rhs_ones) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bv.bv<16>
// CHECK-NEXT:   %rhs_zeros_5 = "smt.eq"(%rhs_zeros_4, %rhs_ones) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bool
// CHECK-NEXT:   %rhs_zeros_6 = "smt.and"(%rhs_zeros_3, %rhs_zeros_5) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %res = "smt.bv.or"(%lhs, %rhs) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bv.bv<16>
// CHECK-NEXT:   %res_zeros = "smt.bv.and"(%lhs_zeros, %rhs_zeros) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bv.bv<16>
// CHECK-NEXT:   %res_ones = "smt.bv.or"(%lhs_ones, %rhs_ones) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bv.bv<16>
// CHECK-NEXT:   %0 = "smt.bv.and"(%res, %res_zeros) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bv.bv<16>
// CHECK-NEXT:   %1 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 16>} : () -> !smt.bv.bv<16>
// CHECK-NEXT:   %2 = "smt.eq"(%0, %1) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bool
// CHECK-NEXT:   %3 = "smt.bv.and"(%res, %res_ones) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bv.bv<16>
// CHECK-NEXT:   %4 = "smt.eq"(%3, %res_ones) : (!smt.bv.bv<16>, !smt.bv.bv<16>) -> !smt.bool
// CHECK-NEXT:   %5 = "smt.and"(%2, %4) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %6 = "smt.not"(%5) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:   %7 = "smt.and"(%lhs_zeros_6, %rhs_zeros_6) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %8 = "smt.and"(%7, %6) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   "smt.assert"(%8) : (!smt.bool) -> ()
// CHECK-NEXT:   "smt.check_sat"() : () -> ()
// CHECK-NEXT: }
