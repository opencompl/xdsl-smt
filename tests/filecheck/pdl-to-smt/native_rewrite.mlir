// RUN: xdsl-smt "%s" -p=pdl-to-smt,lower-effects,canonicalize,dce -t smt | filecheck "%s"

builtin.module {
    pdl.pattern @add_constant_fold : benefit(0) {
        %type = pdl.type : i32

        %c0 = pdl.attribute : %type
        %c1 = pdl.attribute : %type

        %constant0 = pdl.operation "arith.constant" {"value" = %c0} -> (%type : !pdl.type)
        %constant1 = pdl.operation "arith.constant" {"value" = %c1} -> (%type : !pdl.type)

        %lhs = pdl.result 0 of %constant0
        %rhs = pdl.result 0 of %constant1

        %add = pdl.operation "arith.addi" (%lhs, %rhs : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add {
            %res = pdl.apply_native_rewrite "addi"(%c0, %c1 : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %res_constant = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            pdl.replace %add with %res_constant
        }
    }
}
// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (declare-const tmp Bool)
// CHECK-NEXT:  (declare-const c0 Int)
// CHECK-NEXT:  (declare-const c1 Int)
// CHECK-NEXT:  (assert (let ((tmp_0 32))
// CHECK-NEXT:    (= tmp_0 32)))
// CHECK-NEXT:  (declare-fun pow2 ((Int))Int)
// CHECK-NEXT:  (assert (forall ((tmp_0 Int) (tmp_1 Int)) (=> (> tmp_0 tmp_1) (> (pow2 tmp_0) (pow2 tmp_1)))))
// CHECK-NEXT:  (assert (let ((tmp_2 (pow2 32)))
// CHECK-NEXT:    (let ((tmp_3 32))
// CHECK-NEXT:    (let ((tmp_4 (pow2 tmp_3)))
// CHECK-NEXT:    (not (or tmp (and (not tmp) (= (mod (+ (+ c0 c1) tmp_4) tmp_4) (mod (+ (+ c0 c1) tmp_2) tmp_2)))))))))
// CHECK-NEXT:  (check-sat)
