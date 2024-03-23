// RUN: verify-pdl "%s" -opt | filecheck "%s"

// ShrOp(x, cst) -> Concat(zeros, Extract(x)) with cst < width(x) and cst != 0
pdl.pattern @ShrRhsKnownConstant : benefit(0) {
    %i32 = pdl.type : i32
    %type = pdl.type : !transfer.integer

    // Limitation of the current SMT solving capabilities. We cannot express
    // the type i{shift} in a generic way, so we have to duplicate this pattern "for all shift values".
    // Note that this pattern will likely cannot be used to rewrite, as shift_type is never actually matched.
    %shift_type = pdl.type : !transfer.integer

    %shift_attr = pdl.attribute : %type
    %shift_op = pdl.operation "hw.constant" {"value" = %shift_attr} -> (%type : !pdl.type)
    %shift = pdl.result 0 of %shift_op

    // Make sure shift has the same value as the width of shift_type
    pdl.apply_native_constraint "is_equal_to_width_of_type"(%shift_attr, %shift_type : !pdl.attribute, !pdl.type)

    // Check that shift < width(x) using the comparison on the type widths
    pdl.apply_native_constraint "is_greater_integer_type"(%type, %shift_type : !pdl.type, !pdl.type)
    pdl.apply_native_constraint "is_not_zero"(%shift_attr : !pdl.attribute)

    %x = pdl.operand : %type

    %shru_op = pdl.operation "comb.shru"(%x, %shift : !pdl.value, !pdl.value) -> (%type : !pdl.type)

    pdl.rewrite %shru_op {
        %zero_constant = pdl.apply_native_rewrite "get_zero_attr"(%shift_type : !pdl.type) : !pdl.attribute
        %zero_op = pdl.operation "hw.constant" {"value" = %zero_constant} -> (%shift_type : !pdl.type)
        %zero = pdl.result 0 of %zero_op

        %type_width = pdl.apply_native_rewrite "get_width"(%type, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
        %shift_width = pdl.apply_native_rewrite "get_width"(%shift_type, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
        %extract_width = pdl.apply_native_rewrite "subi"(%type_width, %shift_width : !pdl.attribute, !pdl.attribute) : !pdl.attribute
        %extract_type = pdl.apply_native_rewrite "integer_type_from_width"(%extract_width : !pdl.attribute) : !pdl.type

        %extract_op = pdl.operation "comb.extract"(%x : !pdl.value) {"low_bit" = %shift_attr} -> (%extract_type : !pdl.type)
        %extract = pdl.result 0 of %extract_op

        %res_op = pdl.operation "comb.concat"(%zero, %extract : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.replace %shru_op with %res_op
    }
}

// CHECK: All patterns are sound
