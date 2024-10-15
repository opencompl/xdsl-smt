// RUN: xdsl-smt "%s" -p=pdl-to-smt,lower-effects,canonicalize,dce -t smt | filecheck "%s"

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
// CHECK-NEXT:  (declare-const tmp Bool)
// CHECK-NEXT:  (declare-const tmp_0 (Pair (_ BitVec 32) Bool))
// CHECK-NEXT:  (declare-const tmp_1 (Pair (_ BitVec 32) Bool))
// CHECK-NEXT:  (assert (let ((tmp_2 (or (second tmp_0) (second tmp_1))))
// CHECK-NEXT:    (let ((tmp_3 (bvor (first tmp_0) (first tmp_1))))
// CHECK-NEXT:    (not (or tmp (and (not tmp) (=> (not (or (second tmp_0) tmp_2)) (and (= (bvor (first tmp_0) tmp_3) tmp_3) (not tmp_2)))))))))
// CHECK-NEXT:  (check-sat)
