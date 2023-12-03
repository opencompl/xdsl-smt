// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // or extui(x), extui(y) -> extui(or(x,y))
    pdl.pattern @OrIOfExtUI : benefit(0) {
        %type = pdl.type : !transfer.integer
        %new_type = pdl.type : !transfer.integer

        pdl.apply_native_constraint "is_greater_integer_type"(%new_type, %type : !pdl.type, !pdl.type)

        %i64 = pdl.type : i64

        %x = pdl.operand : %type
        %y = pdl.operand : %type

        %extui_x_op = pdl.operation "arith.extui"(%x : !pdl.value) -> (%new_type : !pdl.type)
        %extui_x = pdl.result 0 of %extui_x_op

        %extui_y_op = pdl.operation "arith.extui"(%y : !pdl.value) -> (%new_type : !pdl.type)
        %extui_y = pdl.result 0 of %extui_y_op

        %or_op = pdl.operation "arith.ori"(%extui_x, %extui_y : !pdl.value, !pdl.value) -> (%new_type : !pdl.type)

        pdl.rewrite %or_op {
            %new_or = pdl.operation "arith.ori"(%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            %or = pdl.result 0 of %new_or

            %new_extui = pdl.operation "arith.extui"(%or : !pdl.value) -> (%new_type : !pdl.type)
            pdl.replace %or_op with %new_extui
        }
    }

    // or extsi(x), extsi(y) -> extsi(or(x,y))
    pdl.pattern @OrIOfextsi : benefit(0) {
        %type = pdl.type : !transfer.integer
        %new_type = pdl.type : !transfer.integer

        pdl.apply_native_constraint "is_greater_integer_type"(%new_type, %type : !pdl.type, !pdl.type)

        %i64 = pdl.type : i64

        %x = pdl.operand : %type
        %y = pdl.operand : %type

        %extsi_x_op = pdl.operation "arith.extsi"(%x : !pdl.value) -> (%new_type : !pdl.type)
        %extsi_x = pdl.result 0 of %extsi_x_op

        %extsi_y_op = pdl.operation "arith.extsi"(%y : !pdl.value) -> (%new_type : !pdl.type)
        %extsi_y = pdl.result 0 of %extsi_y_op

        %or_op = pdl.operation "arith.ori"(%extsi_x, %extsi_y : !pdl.value, !pdl.value) -> (%new_type : !pdl.type)

        pdl.rewrite %or_op {
            %new_or = pdl.operation "arith.ori"(%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            %or = pdl.result 0 of %new_or

            %new_extsi = pdl.operation "arith.extsi"(%or : !pdl.value) -> (%new_type : !pdl.type)
            pdl.replace %or_op with %new_extsi
        }
    }
}

// CHECK: All patterns are sound
