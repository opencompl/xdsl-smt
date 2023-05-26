// RUN: xdsl-smt.py %s -t mlir | xdsl-smt.py %s -f mlir -t mlir --print-op-generic | filecheck %s

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


// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "pdl.pattern"() ({
// CHECK-NEXT:     %type = "pdl.type"() {"constantType" = i32} : () -> !pdl.type
// CHECK-NEXT:     %lhs, %lhs_zeros, %lhs_ones = "pdl.kb.operand"(%type) : (!pdl.type) -> (!pdl.value, i32, i32)
// CHECK-NEXT:     %rhs, %rhs_zeros, %rhs_ones = "pdl.kb.operand"(%type) : (!pdl.type) -> (!pdl.value, i32, i32)
// CHECK-NEXT:     %op = "pdl.operation"(%lhs, %rhs, %type) {"opName" = "arith.ori", "attributeValueNames" = [], "operand_segment_sizes" = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
// CHECK-NEXT:     "pdl.rewrite"(%op) ({
// CHECK-NEXT:       %res_zeros = "arith.andi"(%lhs_zeros, %rhs_zeros) : (i32, i32) -> i32
// CHECK-NEXT:       %res_ones = "arith.ori"(%lhs_ones, %rhs_ones) : (i32, i32) -> i32
// CHECK-NEXT:       "pdl.kb.attach"(%op, %res_zeros, %res_ones) : (!pdl.operation, i32, i32) -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 1, 0>} : (!pdl.operation) -> ()
// CHECK-NEXT:   }) {"benefit" = 1 : i16, "sym_name" = "addi_analysis"} : () -> ()
// CHECK-NEXT: }) : () -> ()
