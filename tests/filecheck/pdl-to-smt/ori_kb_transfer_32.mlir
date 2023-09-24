// RUN: xdsl-smt "%s" -t mlir -p=pdl-to-smt,canonicalize-smt | filecheck "%s"


builtin.module {
  pdl.pattern @addi_analysis : benefit(0) {
    // Get an i32 type
    %type = pdl.type : i32

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


// CHECK:       builtin.module {
// CHECK-NEXT:    %lhs = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:    %rhs = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:    %lhs_zeros = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:    %lhs_ones = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:    %lhs_zeros_1 = "smt.utils.first"(%lhs) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %lhs_zeros_2 = "smt.bv.and"(%lhs_zeros_1, %lhs_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %lhs_zeros_3 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:    %lhs_zeros_4 = "smt.eq"(%lhs_zeros_2, %lhs_zeros_3) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:    %lhs_zeros_5 = "smt.bv.and"(%lhs_zeros_1, %lhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %lhs_zeros_6 = "smt.eq"(%lhs_zeros_5, %lhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:    %lhs_zeros_7 = "smt.and"(%lhs_zeros_4, %lhs_zeros_6) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %rhs_zeros = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:    %rhs_ones = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:    %rhs_zeros_1 = "smt.utils.first"(%rhs) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %rhs_zeros_2 = "smt.bv.and"(%rhs_zeros_1, %rhs_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %rhs_zeros_3 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:    %rhs_zeros_4 = "smt.eq"(%rhs_zeros_2, %rhs_zeros_3) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:    %rhs_zeros_5 = "smt.bv.and"(%rhs_zeros_1, %rhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %rhs_zeros_6 = "smt.eq"(%rhs_zeros_5, %rhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:    %rhs_zeros_7 = "smt.and"(%rhs_zeros_4, %rhs_zeros_6) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %0 = "smt.utils.first"(%lhs) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %1 = "smt.utils.first"(%rhs) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %2 = "smt.bv.or"(%0, %1) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %res_zeros = "smt.bv.and"(%lhs_zeros, %rhs_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %res_ones = "smt.bv.or"(%lhs_ones, %rhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %3 = "smt.bv.and"(%2, %res_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %4 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:    %5 = "smt.eq"(%3, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:    %6 = "smt.bv.and"(%2, %res_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:    %7 = "smt.eq"(%6, %res_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:    %8 = "smt.and"(%5, %7) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %9 = "smt.not"(%8) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %10 = "smt.and"(%lhs_zeros_7, %rhs_zeros_7) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %11 = "smt.and"(%10, %9) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    "smt.assert"(%11) : (!smt.bool) -> ()
// CHECK-NEXT:    "smt.check_sat"() : () -> ()
// CHECK-NEXT:  }
