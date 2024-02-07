// RUN: verify-pdl "%s" --max-bitwidth=8 | filecheck "%s"

builtin.module {
    // extsi(extui(x iN : iM) : iL) -> extui(x : iL)
    pdl.pattern @ExtSIOfExtUI : benefit(0) {
        %iN = pdl.type : !transfer.integer
        %iM = pdl.type : !transfer.integer
        %iL = pdl.type : !transfer.integer

        pdl.apply_native_constraint "is_greater_integer_type"(%iL, %iM : !pdl.type, !pdl.type)
        pdl.apply_native_constraint "is_greater_integer_type"(%iM, %iN : !pdl.type, !pdl.type)

        %x = pdl.operand : %iN

        %extsi_x_op = pdl.operation "arith.extui"(%x : !pdl.value) -> (%iM : !pdl.type)
        %extsi_x = pdl.result 0 of %extsi_x_op

        %res_op = pdl.operation "arith.extsi"(%extsi_x : !pdl.value) -> (%iL : !pdl.type)

        pdl.rewrite %res_op {
            %new_res_op = pdl.operation "arith.extui"(%x : !pdl.value) -> (%iL : !pdl.type)
            pdl.replace %res_op with %new_res_op
        }
    }
}

// CHECK: All patterns are sound
