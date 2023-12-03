// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // xor(cmpi(pred, a, b), 1) -> cmpi(~pred, a, b)
    pdl.pattern @XOrINotCmpI : benefit(0) {
        %type = pdl.type : !transfer.integer
        %i64 = pdl.type : i64

        %predicate_attr = pdl.attribute : %i64
        pdl.apply_native_constraint "is_arith_cmpi_predicate"(%predicate_attr : !pdl.attribute)

        %one_attr = pdl.attribute = 1 : i1

        %one_op = pdl.operation "arith.constant" {"value" = %one_attr} -> (%type : !pdl.type)
        %one = pdl.result 0 of %one_op

        %a = pdl.operand : %type
        %b = pdl.operand : %type

        %cmpi_op = pdl.operation "arith.cmpi"(%a, %b : !pdl.value, !pdl.value) {"predicate" = %predicate_attr} -> (%type : !pdl.type)
        %cmpi = pdl.result 0 of %cmpi_op

        %xor_op = pdl.operation "arith.xori"(%cmpi, %one : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %xor_op {
            %inverted_predicate_attr = pdl.apply_native_rewrite "invert_arith_cmpi_predicate"(%predicate_attr : !pdl.attribute) : !pdl.attribute
            %new_cmpi = pdl.operation "arith.cmpi"(%a, %b : !pdl.value, !pdl.value) {"predicate" = %inverted_predicate_attr} -> (%type : !pdl.type)
            pdl.replace %xor_op with %new_cmpi
        }
    }
}

// CHECK-NOT: UNSOUND
