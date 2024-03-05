// RUN: verify-pdl "%s" -opt | filecheck "%s"

// Missing:
// `and(replicate(x), powerOfTwCst)` -> `concat(zeros, x, zeros)` for `x : i1`, and powerOfTwoCst != 1 and != 2^(n - 1)
// `and(extract(x), 0000???00)` -> `concat(0000, and(extract(x), ???), 00)` with a smaller extract

// and(x, and(val1, val2)) -> and(x, val1, val2) -- flatten
pdl.pattern @AndFlatten : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t
    %val1 = pdl.operand : %t
    %val2 = pdl.operand : %t

    %inner_and_op = pdl.operation "comb.and"(%val1, %val2 : !pdl.value, !pdl.value) -> (%t: !pdl.type)
    %inner_and = pdl.result 0 of %inner_and_op

    %and_op = pdl.operation "comb.and"(%x, %inner_and : !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %and_op {
        %new_op = pdl.operation "comb.and"(%x, %val1, %val2 : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %and_op with %new_op
    }
}

// and(x, val1, x) -> and(x, val1) -- idempotent
pdl.pattern @AndIdempotent : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t
    %val1 = pdl.operand : %t

    %and_op = pdl.operation "comb.and"(%x, %val1, %x : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %and_op {
        %new_op = pdl.operation "comb.and"(%x, %val1: !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %and_op with %new_op
    }
}

// and(x, -1, y) -> and(x, y)
pdl.pattern @AndMinusOne : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t
    %y = pdl.operand : %t

    %cst_attr = pdl.attribute : %t
    pdl.apply_native_constraint "is_minus_one"(%cst_attr : !pdl.attribute)

    %cst_op = pdl.operation "hw.constant" {"value" = %cst_attr} -> (%t: !pdl.type)
    %cst = pdl.result 0 of %cst_op

    %and_op = pdl.operation "comb.and"(%x, %cst, %y : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %and_op {
        %new_op = pdl.operation "comb.and"(%x, %y: !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %and_op with %new_op
    }
}

// and(x, -1) -> x
pdl.pattern @AndMinusOne : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t

    %cst_attr = pdl.attribute : %t
    pdl.apply_native_constraint "is_minus_one"(%cst_attr : !pdl.attribute)

    %cst_op = pdl.operation "hw.constant" {"value" = %cst_attr} -> (%t: !pdl.type)
    %cst = pdl.result 0 of %cst_op

    %and_op = pdl.operation "comb.and"(%x, %cst : !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %and_op {
        pdl.replace %and_op with (%x : !pdl.value)
    }
}

// and(x, cst1, cst2) -> and(x, cst1 & cst2)
pdl.pattern @AndMinusOne : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t

    %cst1_attr = pdl.attribute : %t
    %cst1_op = pdl.operation "hw.constant" {"value" = %cst1_attr} -> (%t: !pdl.type)
    %cst1 = pdl.result 0 of %cst1_op

    %cst2_attr = pdl.attribute : %t
    %cst2_op = pdl.operation "hw.constant" {"value" = %cst2_attr} -> (%t: !pdl.type)
    %cst2 = pdl.result 0 of %cst2_op

    %and_op = pdl.operation "comb.and"(%x, %cst1, %cst2 : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %and_op {
        %merged_cst = pdl.apply_native_rewrite "andi"(%cst1_attr, %cst2_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
        %cst_op = pdl.operation "hw.constant" {"value" = %merged_cst} -> (%t: !pdl.type)
        %cst = pdl.result 0 of %cst_op
        %new_op = pdl.operation "comb.and"(%x, %cst: !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %and_op with %new_op
    }
}

// `and(replicate(x), 1)` -> `concat(zeros, x)` for `x : i1`
pdl.pattern @AndReplicateOne : benefit(0) {
    %i1 = pdl.type : i1
    %t = pdl.type : !transfer.integer
    pdl.apply_native_constraint "is_greater_integer_type"(%t, %i1 : !pdl.type, !pdl.type)
    %t_minus_one = pdl.apply_native_rewrite "integer_type_sub_width"(%t, %i1 : !pdl.type, !pdl.type) : !pdl.type

    %x = pdl.operand : %i1
    %replicate_op = pdl.operation "comb.replicate"(%x : !pdl.value) -> (%t: !pdl.type)
    %replicate = pdl.result 0 of %replicate_op

    %one_attr = pdl.apply_native_rewrite "get_one_attr"(%t : !pdl.type) : !pdl.attribute
    %one_op = pdl.operation "hw.constant" {"value" = %one_attr} -> (%t: !pdl.type)
    %one = pdl.result 0 of %one_op

    %and_op = pdl.operation "comb.and"(%replicate, %one : !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %and_op {
        %zero_attr = pdl.apply_native_rewrite "get_zero_attr"(%t_minus_one : !pdl.type) : !pdl.attribute
        %zero_op = pdl.operation "hw.constant" {"value" = %zero_attr} -> (%t: !pdl.type)
        %zero = pdl.result 0 of %zero_op
        %new_op = pdl.operation "comb.concat"(%zero, %x : !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %and_op with %new_op
    }
}

// `and(replicate(x), 2^(n - 1))` -> `concat(x, zeros)` for `x : i1`
pdl.pattern @AndReplicateMin : benefit(0) {
    %i1 = pdl.type : i1
    %t = pdl.type : !transfer.integer
    pdl.apply_native_constraint "is_greater_integer_type"(%t, %i1 : !pdl.type, !pdl.type)
    %t_minus_one = pdl.apply_native_rewrite "integer_type_sub_width"(%t, %i1 : !pdl.type, !pdl.type) : !pdl.type

    %x = pdl.operand : %i1
    %replicate_op = pdl.operation "comb.replicate"(%x : !pdl.value) -> (%t: !pdl.type)
    %replicate = pdl.result 0 of %replicate_op

    %cst_attr = pdl.apply_native_rewrite "get_minimum_signed_value"(%t : !pdl.type) : !pdl.attribute
    %cst_op = pdl.operation "hw.constant" {"value" = %cst_attr} -> (%t: !pdl.type)
    %cst = pdl.result 0 of %cst_op

    %and_op = pdl.operation "comb.and"(%replicate, %cst : !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %and_op {
        %zero_attr = pdl.apply_native_rewrite "get_zero_attr"(%t_minus_one : !pdl.type) : !pdl.attribute
        %zero_op = pdl.operation "hw.constant" {"value" = %zero_attr} -> (%t: !pdl.type)
        %zero = pdl.result 0 of %zero_op
        %new_op = pdl.operation "comb.concat"(%x, %zero : !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %and_op with %new_op
    }
}

// CHECK: All patterns are sound
