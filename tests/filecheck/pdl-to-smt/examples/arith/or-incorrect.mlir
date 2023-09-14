// RUN: xdsl-smt "%s" -p=pdl-to-smt -t smt | z3 -in | filecheck "%s"

// or(x, or(x, y)) -> or(x, y)

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %2 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %3 = "pdl.operation"(%1, %2, %0) {attributeValueNames = [], opName = "arith.ori", operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%3) ({
      "pdl.replace"(%3, %2) {operand_segment_sizes = array<i32: 1, 0, 1>} : (!pdl.operation, !pdl.value) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16} : () -> ()
}) : () -> ()


// CHECK-NOT: unsat
