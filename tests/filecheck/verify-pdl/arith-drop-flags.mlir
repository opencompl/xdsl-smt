// RUN: verify-pdl "%s" | filecheck "%s"

builtin.module {
    pdl.pattern @drop_overflows : benefit(0) {
        %type = pdl.type : i32

        %lhs = pdl.operand : %type
        %rhs = pdl.operand : %type
        %overflow = pdl.attribute = #arith.overflow<nsw, nuw>

        %add = pdl.operation "arith.addi" (%lhs, %rhs : !pdl.value, !pdl.value) {"overflowFlags" = %overflow} -> (%type : !pdl.type)

        pdl.rewrite %add {
            %no_overflow = pdl.attribute = #arith.overflow<none>
            %add_no_overflow = pdl.operation "arith.addi" (%lhs, %rhs : !pdl.value, !pdl.value) {"overflowFlags" = %no_overflow} -> (%type : !pdl.type)
            pdl.replace %add with %add_no_overflow
        }
    }
}

// CHECK: All patterns are sound
