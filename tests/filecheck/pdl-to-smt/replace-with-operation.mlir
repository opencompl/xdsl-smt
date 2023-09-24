// RUN: xdsl-smt "%s" -p=pdl-to-smt -t smt | filecheck "%s"

// or(x, y) -> or(y, x)

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %2 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %3 = "pdl.operation"(%1, %2, %0) {attributeValueNames = [], opName = "arith.ori", operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
    "pdl.rewrite"(%3) ({
      %5 = "pdl.operation"(%2, %1, %0) {attributeValueNames = [], opName = "arith.ori", operand_segment_sizes = array<i32: 2, 0, 1>} : (!pdl.value, !pdl.value, !pdl.type) -> !pdl.operation
      "pdl.replace"(%3, %5) {operand_segment_sizes = array<i32: 1, 1, 0>} : (!pdl.operation, !pdl.operation) -> ()
    }) {operand_segment_sizes = array<i32: 1, 0>} : (!pdl.operation) -> ()
  }) {benefit = 1 : i16} : () -> ()
}) : () -> ()


// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (declare-const tmp (Pair (_ BitVec 32) Bool))
// CHECK-NEXT:  (declare-const tmp_0 (Pair (_ BitVec 32) Bool))
// CHECK-NEXT:  (assert (distinct (pair (bvor (first tmp_0) (first tmp)) (or (second tmp_0) (second tmp))) (pair (bvor (first tmp) (first tmp_0)) (or (second tmp) (second tmp_0)))))
// CHECK-NEXT:  (check-sat)
