// RUN: verify-pdl "%s" | filecheck "%s"

builtin.module {
  pdl.pattern @addi_analysis : benefit(0) {
    // Get an i16 type
    %type = pdl.type : !transfer.integer

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

// CHECK:      with types (1,): SOUND
// CHECK-NEXT: with types (2,): SOUND
// CHECK-NEXT: with types (3,): SOUND
// CHECK-NEXT: with types (4,): SOUND
// CHECK-NEXT: with types (5,): SOUND
// CHECK-NEXT: with types (6,): SOUND
// CHECK-NEXT: with types (7,): SOUND
// CHECK-NEXT: with types (8,): SOUND
// CHECK-NEXT: with types (9,): SOUND
// CHECK-NEXT: with types (10,): SOUND
// CHECK-NEXT: with types (11,): SOUND
// CHECK-NEXT: with types (12,): SOUND
// CHECK-NEXT: with types (13,): SOUND
// CHECK-NEXT: with types (14,): SOUND
// CHECK-NEXT: with types (15,): SOUND
// CHECK-NEXT: with types (16,): SOUND
// CHECK-NEXT: with types (17,): SOUND
// CHECK-NEXT: with types (18,): SOUND
// CHECK-NEXT: with types (19,): SOUND
// CHECK-NEXT: with types (20,): SOUND
// CHECK-NEXT: with types (21,): SOUND
// CHECK-NEXT: with types (22,): SOUND
// CHECK-NEXT: with types (23,): SOUND
// CHECK-NEXT: with types (24,): SOUND
// CHECK-NEXT: with types (25,): SOUND
// CHECK-NEXT: with types (26,): SOUND
// CHECK-NEXT: with types (27,): SOUND
// CHECK-NEXT: with types (28,): SOUND
// CHECK-NEXT: with types (29,): SOUND
// CHECK-NEXT: with types (30,): SOUND
// CHECK-NEXT: with types (31,): SOUND
// CHECK-NEXT: with types (32,): SOUND
