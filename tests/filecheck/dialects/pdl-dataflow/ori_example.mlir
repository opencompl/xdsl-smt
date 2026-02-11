// RUN: xdsl-smt "%s" | filecheck "%s"

builtin.module {
  pdl.pattern @addi_analysis : benefit(0) {
    // Get an i32 type
    %type = pdl.type : i32

    // Get the two operation operands
    %lhs = pdl.operand : %type
    %rhs = pdl.operand : %type

    // Get the operands analysis values
    %lhs_zeros, %lhs_ones = "pdl.dataflow.get"(%lhs) {domain_name = "kb"} : (!pdl.value) -> (!transfer.integer<8>, !transfer.integer<8>)
    %rhs_zeros, %rhs_ones = "pdl.dataflow.get"(%rhs) {domain_name = "kb"} : (!pdl.value) -> (!transfer.integer<8>, !transfer.integer<8>)

    // Get an add operation that takes both operands
    %op = pdl.operation "arith.ori"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%type : !pdl.type)

    pdl.rewrite %op {
        // Compute the known bits of the result
        %res_zeros = "transfer.and"(%lhs_zeros, %rhs_zeros) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
        %res_ones = "transfer.or"(%lhs_ones, %rhs_ones) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>

        // Get the value of the operation result
        %res = pdl.result 0 of %op

        // Attach it to the operation
        "pdl.dataflow.attach"(%res, %res_zeros, %res_ones) {domain_name = "kb"} : (!pdl.value, !transfer.integer<8>, !transfer.integer<8>) -> ()
    }
  }
}


// CHECK:      builtin.module {
// CHECK-NEXT:   pdl.pattern @addi_analysis : benefit(0) {
// CHECK-NEXT:     %type = pdl.type : i32
// CHECK-NEXT:     %lhs = pdl.operand : %type
// CHECK-NEXT:     %rhs = pdl.operand : %type
// CHECK-NEXT:     %lhs_zeros, %lhs_ones = "pdl.dataflow.get"(%lhs) {domain_name = "kb"} : (!pdl.value) -> (!transfer.integer<8>, !transfer.integer<8>)
// CHECK-NEXT:     %rhs_zeros, %rhs_ones = "pdl.dataflow.get"(%rhs) {domain_name = "kb"} : (!pdl.value) -> (!transfer.integer<8>, !transfer.integer<8>)
// CHECK-NEXT:     %op = pdl.operation "arith.ori" (%lhs, %rhs : !pdl.value, !pdl.value) -> (%type : !pdl.type)
// CHECK-NEXT:     pdl.rewrite %op {
// CHECK-NEXT:       %res_zeros = "transfer.and"(%lhs_zeros, %rhs_zeros) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
// CHECK-NEXT:       %res_ones = "transfer.or"(%lhs_ones, %rhs_ones) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
// CHECK-NEXT:       %res = pdl.result 0 of %op
// CHECK-NEXT:       "pdl.dataflow.attach"(%res, %res_zeros, %res_ones) {domain_name = "kb"} : (!pdl.value, !transfer.integer<8>, !transfer.integer<8>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
