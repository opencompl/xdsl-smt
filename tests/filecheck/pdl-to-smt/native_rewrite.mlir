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
// CHECK-NEXT:  (declare-const c0 Int)
// CHECK-NEXT:  (declare-const c1 Int)
// CHECK-NEXT:  (assert (let ((tmp 32))
// CHECK-NEXT:    (= tmp 32)))
// CHECK-NEXT:  (declare-fun pow2 ((Int))Int)
// CHECK-NEXT:  (assert (forall ((tmp Int) (tmp_0 Int)) (=> (> tmp tmp_0) (> (pow2 tmp) (pow2 tmp_0)))))
// CHECK-NEXT:  (assert (let ((tmp_1 (pow2 32)))
// CHECK-NEXT:    (let ((tmp_2 32))
// CHECK-NEXT:    (let ((tmp_3 (pow2 tmp_2)))
// CHECK-NEXT:    (not (= (mod (+ (+ c0 c1) tmp_3) tmp_3) (mod (+ (+ c0 c1) tmp_1) tmp_1)))))))
// CHECK-NEXT:  (check-sat)
