// RUN: xdsl-smt "%s" -p=pdl-to-smt,canonicalize-smt -t smt | filecheck "%s"

// or(x, or(x, y)) -> or(x, y)

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %2 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %3 = pdl.operation "arith.ori"(%1, %2 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
    %4 = "pdl.result"(%3) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    %5 = pdl.operation "arith.ori"(%1, %4 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
    pdl.rewrite %5 {
      pdl.replace %5 with (%4 : !pdl.value)
    }
  }) {benefit = 1 : i16} : () -> ()
}) : () -> ()

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (declare-const tmp (Pair (_ BitVec 32) Bool))
// CHECK-NEXT:  (declare-const tmp_0 (Pair (_ BitVec 32) Bool))
// CHECK-NEXT:  (assert (let ((tmp_1 (or (second tmp) (second tmp_0))))
// CHECK-NEXT:    (let ((tmp_2 (bvor (first tmp) (first tmp_0))))
// CHECK-NEXT:    (not (=> (not (or (second tmp) tmp_1)) (and (= (bvor (first tmp) tmp_2) tmp_2) (not tmp_1)))))))
// CHECK-NEXT:  (check-sat)
