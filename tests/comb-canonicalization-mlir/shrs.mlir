// RUN: verify-pdl "%s" -opt | filecheck "%s"

// ShrSOp(x, cst) -> Concat(replicate(extract(x, topbit)),extract(x))
pdl.pattern @ShrRhsKnownConstant : benefit(0) {
    %i32 = pdl.type : i32
    %type = pdl.type : !transfer.integer<8>

    // Limitation of the current SMT solving capabilities. We cannot express
    // the type i{shift} in a generic way, so we have to duplicate this pattern "for all shift values".
    // Note that this pattern will likely cannot be used to rewrite, as shift_type is never actually matched.
    %shift_type = pdl.type : !transfer.integer<8>

    %shift_attr = pdl.attribute : %type
    %shift_op = pdl.operation "hw.constant" {"value" = %shift_attr} -> (%type : !pdl.type)
    %shift = pdl.result 0 of %shift_op

    // Make sure shift has the same value as the width of shift_type
    pdl.apply_native_constraint "is_equal_to_width_of_type"(%shift_attr, %shift_type : !pdl.attribute, !pdl.type)

    // Check that shift < width(x) using the comparison on the type widths
    pdl.apply_native_constraint "is_greater_integer_type"(%type, %shift_type : !pdl.type, !pdl.type)
    pdl.apply_native_constraint "is_not_zero"(%shift_attr : !pdl.attribute)

    %x = pdl.operand : %type

    %shrs_op = pdl.operation "comb.shrs"(%x, %shift : !pdl.value, !pdl.value) -> (%type : !pdl.type)

    pdl.rewrite %shrs_op {
        %width = pdl.apply_native_rewrite "get_width"(%type, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
        %one = pdl.apply_native_rewrite "get_one_attr"(%i32 : !pdl.type) : !pdl.attribute
        %last_bit_attr = pdl.apply_native_rewrite "subi"(%width, %one : !pdl.attribute, !pdl.attribute) : !pdl.attribute

        %i1 = pdl.type : i1
        %last_bit_op = pdl.operation "comb.extract"(%x : !pdl.value) {"lowBit" = %last_bit_attr} -> (%i1 : !pdl.type)
        %last_bit = pdl.result 0 of %last_bit_op

        %replicate_op = pdl.operation "comb.replicate"(%last_bit : !pdl.value) -> (%shift_type : !pdl.type)
        %replicate = pdl.result 0 of %replicate_op

        %type_width = pdl.apply_native_rewrite "get_width"(%type, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
        %shift_width = pdl.apply_native_rewrite "get_width"(%shift_type, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
        %extract_width = pdl.apply_native_rewrite "subi"(%type_width, %shift_width : !pdl.attribute, !pdl.attribute) : !pdl.attribute
        %extract_type = pdl.apply_native_rewrite "integer_type_from_width"(%extract_width : !pdl.attribute) : !pdl.type

        %extract_op = pdl.operation "comb.extract"(%x : !pdl.value) {"lowBit" = %shift_attr} -> (%extract_type : !pdl.type)
        %extract = pdl.result 0 of %extract_op

        %res_op = pdl.operation "comb.concat"(%replicate, %extract : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.replace %shrs_op with %res_op
    }
}

// CHECK: All patterns are sound
