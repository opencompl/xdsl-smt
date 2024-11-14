// RUN: verify-pdl "%s" -max-bitwidth=8 -opt | filecheck "%s"

builtin.module {
    // trunci(extsi(x)) -> extsi(x), when only the sign-extension bits are truncated
    pdl.pattern @TruncIExtSIToExtSI : benefit(0) {
        %iN = pdl.type : !transfer.integer
        %iM = pdl.type : !transfer.integer
        %iL = pdl.type : !transfer.integer

        pdl.apply_native_constraint "is_greater_integer_type"(%iN, %iM : !pdl.type, !pdl.type)
        pdl.apply_native_constraint "is_greater_integer_type"(%iM, %iL : !pdl.type, !pdl.type)

        %x = pdl.operand : %iL

        %extsi_x_op = pdl.operation "arith.extsi"(%x : !pdl.value) -> (%iN : !pdl.type)
        %extsi_x = pdl.result 0 of %extsi_x_op

        %trunci_op = pdl.operation "arith.trunci"(%extsi_x : !pdl.value) -> (%iM : !pdl.type)

        pdl.rewrite %trunci_op {
            %new_extsi = pdl.operation "arith.extsi"(%x : !pdl.value) -> (%iM : !pdl.type)
            pdl.replace %trunci_op with %new_extsi
        }
    }

    // trunci(extui(x)) -> extui(x), when only the zero-extension bits are truncated
    pdl.pattern @TruncIExtUIToExtUI : benefit(0) {
        %iN = pdl.type : !transfer.integer
        %iM = pdl.type : !transfer.integer
        %iL = pdl.type : !transfer.integer

        pdl.apply_native_constraint "is_greater_integer_type"(%iN, %iM : !pdl.type, !pdl.type)
        pdl.apply_native_constraint "is_greater_integer_type"(%iM, %iL : !pdl.type, !pdl.type)

        %x = pdl.operand : %iL

        %extui_x_op = pdl.operation "arith.extui"(%x : !pdl.value) -> (%iN : !pdl.type)
        %extui_x = pdl.result 0 of %extui_x_op

        %trunci_op = pdl.operation "arith.trunci"(%extui_x : !pdl.value) -> (%iM : !pdl.type)

        pdl.rewrite %trunci_op {
            %new_extui = pdl.operation "arith.extui"(%x : !pdl.value) -> (%iM : !pdl.type)
            pdl.replace %trunci_op with %new_extui
        }
    }

    // trunci(shrui(mul(sext(x), sext(y)), c)) -> mulsi_extended(x, y)
    // TODO, no semantics for mulsi_extended yet

    // trunci(shrui(mul(zext(x), zext(y)), c)) -> mului_extended(x, y)
    // TODO, no semantics for mului_extended yet
}

// CHECK: All patterns are sound
