// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // cmpi(==, a extsi iNN, b extsi iNN) -> cmpi(==, a, b)
    pdl.pattern @CmpIExtSIEq : benefit(0) {
        %type = pdl.type : !transfer.integer<8>
        %new_type = pdl.type : !transfer.integer<8>

        pdl.apply_native_constraint "is_greater_integer_type"(%new_type, %type : !pdl.type, !pdl.type)

        %a = pdl.operand : %type
        %b = pdl.operand : %type

        %eq_predicate = pdl.attribute = 0 : i64

        %extsi_a_op = pdl.operation "arith.extsi"(%a : !pdl.value) -> (%new_type : !pdl.type)
        %extsi_a = pdl.result 0 of %extsi_a_op

        %extsi_b_op = pdl.operation "arith.extsi"(%b : !pdl.value) -> (%new_type : !pdl.type)
        %extsi_b = pdl.result 0 of %extsi_b_op

        %cmpi_op = pdl.operation "arith.cmpi"(%extsi_a, %extsi_b : !pdl.value, !pdl.value) {"predicate" = %eq_predicate} -> (%new_type : !pdl.type)

        pdl.rewrite %cmpi_op {
            %new_cmpi = pdl.operation "arith.cmpi"(%a, %b : !pdl.value, !pdl.value) {"predicate" = %eq_predicate} -> (%type : !pdl.type)
            pdl.replace %cmpi_op with %new_cmpi
        }
    }

    // cmpi(!=, a extsi iNN, b extsi iNN) -> cmpi(!=, a, b)
    pdl.pattern @CmpIExtSINe : benefit(0) {
        %type = pdl.type : !transfer.integer<8>
        %new_type = pdl.type : !transfer.integer<8>

        pdl.apply_native_constraint "is_greater_integer_type"(%new_type, %type : !pdl.type, !pdl.type)

        %a = pdl.operand : %type
        %b = pdl.operand : %type

        %eq_predicate = pdl.attribute = 1 : i64

        %extsi_a_op = pdl.operation "arith.extsi"(%a : !pdl.value) -> (%new_type : !pdl.type)
        %extsi_a = pdl.result 0 of %extsi_a_op

        %extsi_b_op = pdl.operation "arith.extsi"(%b : !pdl.value) -> (%new_type : !pdl.type)
        %extsi_b = pdl.result 0 of %extsi_b_op

        %cmpi_op = pdl.operation "arith.cmpi"(%extsi_a, %extsi_b : !pdl.value, !pdl.value) {"predicate" = %eq_predicate} -> (%new_type : !pdl.type)

        pdl.rewrite %cmpi_op {
            %new_cmpi = pdl.operation "arith.cmpi"(%a, %b : !pdl.value, !pdl.value) {"predicate" = %eq_predicate} -> (%type : !pdl.type)
            pdl.replace %cmpi_op with %new_cmpi
        }
    }

    // cmpi(==, a extui iNN, b extui iNN) -> cmpi(==, a, b)
    pdl.pattern @CmpIExtUIEq : benefit(0) {
        %type = pdl.type : !transfer.integer<8>
        %new_type = pdl.type : !transfer.integer<8>

        pdl.apply_native_constraint "is_greater_integer_type"(%new_type, %type : !pdl.type, !pdl.type)

        %a = pdl.operand : %type
        %b = pdl.operand : %type

        %eq_predicate = pdl.attribute = 0 : i64

        %extui_a_op = pdl.operation "arith.extui"(%a : !pdl.value) -> (%new_type : !pdl.type)
        %extui_a = pdl.result 0 of %extui_a_op

        %extui_b_op = pdl.operation "arith.extui"(%b : !pdl.value) -> (%new_type : !pdl.type)
        %extui_b = pdl.result 0 of %extui_b_op

        %cmpi_op = pdl.operation "arith.cmpi"(%extui_a, %extui_b : !pdl.value, !pdl.value) {"predicate" = %eq_predicate} -> (%new_type : !pdl.type)

        pdl.rewrite %cmpi_op {
            %new_cmpi = pdl.operation "arith.cmpi"(%a, %b : !pdl.value, !pdl.value) {"predicate" = %eq_predicate} -> (%type : !pdl.type)
            pdl.replace %cmpi_op with %new_cmpi
        }
    }

    // cmpi(!=, a extui iNN, b extui iNN) -> cmpi(!=, a, b)
    pdl.pattern @CmpIExtUINe : benefit(0) {
        %type = pdl.type : !transfer.integer<8>
        %new_type = pdl.type : !transfer.integer<8>

        pdl.apply_native_constraint "is_greater_integer_type"(%new_type, %type : !pdl.type, !pdl.type)

        %a = pdl.operand : %type
        %b = pdl.operand : %type

        %eq_predicate = pdl.attribute = 1 : i64

        %extui_a_op = pdl.operation "arith.extui"(%a : !pdl.value) -> (%new_type : !pdl.type)
        %extui_a = pdl.result 0 of %extui_a_op

        %extui_b_op = pdl.operation "arith.extui"(%b : !pdl.value) -> (%new_type : !pdl.type)
        %extui_b = pdl.result 0 of %extui_b_op

        %cmpi_op = pdl.operation "arith.cmpi"(%extui_a, %extui_b : !pdl.value, !pdl.value) {"predicate" = %eq_predicate} -> (%new_type : !pdl.type)

        pdl.rewrite %cmpi_op {
            %new_cmpi = pdl.operation "arith.cmpi"(%a, %b : !pdl.value, !pdl.value) {"predicate" = %eq_predicate} -> (%type : !pdl.type)
            pdl.replace %cmpi_op with %new_cmpi
        }
    }
}

// CHECK-NOT: UNSOUND
