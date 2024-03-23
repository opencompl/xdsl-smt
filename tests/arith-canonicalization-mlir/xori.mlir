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

    // xor extui(x), extui(y) -> extui(xor(x,y))
    pdl.pattern @XOrIOfExtUI : benefit(0) {
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

        %xor_op = pdl.operation "arith.xori"(%extui_x, %extui_y : !pdl.value, !pdl.value) -> (%new_type : !pdl.type)

        pdl.rewrite %xor_op {
            %new_xor = pdl.operation "arith.xori"(%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            %xor = pdl.result 0 of %new_xor

            %new_extui = pdl.operation "arith.extui"(%xor : !pdl.value) -> (%new_type : !pdl.type)
            pdl.replace %xor_op with %new_extui
        }
    }

    // xor extsi(x), extsi(y) -> extsi(xor(x,y))
    pdl.pattern @XOrIOfExtSI : benefit(0) {
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

        %xor_op = pdl.operation "arith.xori"(%extsi_x, %extsi_y : !pdl.value, !pdl.value) -> (%new_type : !pdl.type)

        pdl.rewrite %xor_op {
            %new_xor = pdl.operation "arith.xori"(%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            %xor = pdl.result 0 of %new_xor

            %new_extsi = pdl.operation "arith.extsi"(%xor : !pdl.value) -> (%new_type : !pdl.type)
            pdl.replace %xor_op with %new_extsi
        }
    }
}

// CHECK-NOT: UNSOUND
