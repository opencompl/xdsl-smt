// RUN: xdsl-smt "%s" -p=pdl-to-smt,canonicalize-smt -t smt | filecheck "%s"

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

// CHECK:       (declare-const c0 (_ BitVec 32))
// CHECK-NEXT:  (declare-const c1 (_ BitVec 32))
// CHECK-NEXT:  (assert (distinct (pair (bvadd c0 c1) false) (pair (bvadd c0 c1) false)))
// CHECK-NEXT:  (check-sat)
