// RUN: xdsl-smt.py %s -t mlir | xdsl-smt.py %s -f mlir -t mlir | filecheck %s

"builtin.module"() ({
  "pdl.pattern"() ({
    // Get an i32 type
    %type = "pdl.type"() {"constantType" = i32} : () -> !pdl.type

    // Get two i32 values that have a known bits pattern
    %lhs, %lhs_kb = "pdl.kb.operand"(%type) : (!pdl.type) -> (!pdl.value, !pdl.attribute)
    %rhs, %rhs_kb = "pdl.kb.operand"(%type) : (!pdl.type) -> (!pdl.value, !pdl.attribute)

    // Get an add operation that takes both operands
    %op = "pdl.operation"(%lhs, %rhs, %type) {opName = "arith.addi", attributeValueNames = [], operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation

    "pdl.rewrite"(%op) ({
        // Compute the known bits of the result
        %res_kb = "pdl.kb.add"(%lhs_kb, %rhs_kb) : (!pdl.attribute, !pdl.attribute) -> !pdl.attribute

        // Attach it to the operation
        "pdl.kb.attach"(%op, %res_kb) : (!pdl.operation, !pdl.attribute) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16, sym_name = "addi_analysis"} : () -> ()
}) : () -> ()


// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "pdl.pattern"() ({
// CHECK-NEXT:     %type = "pdl.type"() {"constantType" = i32} : () -> !pdl.type
// CHECK-NEXT:     %lhs, %lhs_kb = "pdl.kb.operand"(%type) : (!pdl.type) -> (!pdl.value, !pdl.attribute)
// CHECK-NEXT:     %rhs, %rhs_kb = "pdl.kb.operand"(%type) : (!pdl.type) -> (!pdl.value, !pdl.attribute)
// CHECK-NEXT:     %op = "pdl.operation"(%lhs, %rhs, %type) {"opName" = "arith.addi", "attributeValueNames" = [], "operand_segment_sizes" = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
// CHECK-NEXT:     "pdl.rewrite"(%op) ({
// CHECK-NEXT:       %res_kb = "pdl.kb.add"(%lhs_kb, %rhs_kb) : (!pdl.attribute, !pdl.attribute) -> !pdl.attribute
// CHECK-NEXT:       "pdl.kb.attach"(%op, %res_kb) : (!pdl.operation, !pdl.attribute) -> ()
// CHECK-NEXT:     }) {"operand_segment_sizes" = array<i32: 1, 0>} : (!pdl.operation) -> ()
// CHECK-NEXT:   }) {"benefit" = 1 : i16, "sym_name" = "addi_analysis"} : () -> ()
// CHECK-NEXT: }) : () -> ()
