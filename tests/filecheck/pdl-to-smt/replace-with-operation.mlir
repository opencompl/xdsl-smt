// RUN: xdsl-smt.py %s | filecheck %s

// or(x, y) -> or(y, x)

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %2 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %3 = "pdl.operation"(%1, %2, %0) {attributeValueNames = [], opName = "arith.ori", operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%3) ({
      %5 = "pdl.operation"(%2, %1, %0) {attributeValueNames = [], opName = "arith.ori", operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
      "pdl.replace"(%3, %5) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.value) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16} : () -> ()
}) : () -> ()


// CHECK:      (declare-const tmp (_ BitVec 32))
// CHECK-NEXT: (declare-const tmp_0 (_ BitVec 32))
// CHECK-NEXT: (assert (distinct (bvor tmp_0 tmp) (bvor tmp tmp_0)))
// CHECK-NEXT: (check-sat)
