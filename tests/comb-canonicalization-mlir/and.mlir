// RUN: verify-pdl "%s" -opt | filecheck "%s"

// Missing:
// `and(replicate(x), powerOfTwCst)` -> `concat(zeros, x, zeros)` for `x : i1`, and powerOfTwoCst != 1 and != 2^(n - 1)
// `and(extract(x), 0000???00)` -> `concat(0000, and(extract(x), ???), 00)` with a smaller extract
// extracts only of and(...) -> and(extract()...)

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
    %i32 = pdl.type : i32
    %t = pdl.type : !transfer.integer
    pdl.apply_native_constraint "is_greater_integer_type"(%t, %i1 : !pdl.type, !pdl.type)

    %type_width = pdl.apply_native_rewrite "get_width"(%t, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
    %1 = pdl.attribute = 1 : i32
    %t_minus_one_width = pdl.apply_native_rewrite "subi"(%type_width, %1 : !pdl.attribute, !pdl.attribute) : !pdl.attribute
    %t_minus_one = pdl.apply_native_rewrite "integer_type_from_width"(%t_minus_one_width : !pdl.attribute) : !pdl.type

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
    %i32 = pdl.type : i32
    %t = pdl.type : !transfer.integer
    pdl.apply_native_constraint "is_greater_integer_type"(%t, %i1 : !pdl.type, !pdl.type)

    %type_width = pdl.apply_native_rewrite "get_width"(%t, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
    %1 = pdl.attribute = 1 : i32
    %t_minus_one_width = pdl.apply_native_rewrite "subi"(%type_width, %1 : !pdl.attribute, !pdl.attribute) : !pdl.attribute
    %t_minus_one = pdl.apply_native_rewrite "integer_type_from_width"(%t_minus_one_width : !pdl.attribute) : !pdl.type

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

// and(concat(x, cst1), a, b, c, cst2)
//    ==> and(a, b, c, concat(and(x,cst2'), and(cst1,cst2'')).
pdl.pattern @AndConcatCst : benefit(0) {
    %i32 = pdl.type : i32
    %t = pdl.type : !transfer.integer
    %t_left = pdl.type : !transfer.integer

    pdl.apply_native_constraint "is_greater_integer_type"(%t, %t_left : !pdl.type, !pdl.type)

    %t_width = pdl.apply_native_rewrite "get_width"(%t, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
    %t_left_width = pdl.apply_native_rewrite "get_width"(%t_left, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
    %t_right_width = pdl.apply_native_rewrite "subi"(%t_width, %t_left_width : !pdl.attribute, !pdl.attribute) : !pdl.attribute
    %t_right = pdl.apply_native_rewrite "integer_type_from_width"(%t_right_width : !pdl.attribute) : !pdl.type

    %x = pdl.operand : %t_left

    %cst1_attr = pdl.attribute : %t_right
    %cst1_op = pdl.operation "hw.constant" {"value" = %cst1_attr} -> (%t_right : !pdl.type)
    %cst1 = pdl.result 0 of %cst1_op

    %concat_op = pdl.operation "comb.concat"(%x, %cst1 : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    %concat = pdl.result 0 of %concat_op

    %a = pdl.operand : %t
    %b = pdl.operand : %t
    %c = pdl.operand : %t

    %cst2_attr = pdl.attribute : %t
    %cst2_op = pdl.operation "hw.constant" {"value" = %cst2_attr} -> (%t : !pdl.type)
    %cst2 = pdl.result 0 of %cst2_op

    %and_op = pdl.operation "comb.and"(%concat, %a, %b, %c, %cst2 : !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %and_op {
        %right_width = pdl.apply_native_rewrite "get_width"(%t_right, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
        %cst2_prime_op = pdl.operation "comb.extract"(%cst2 : !pdl.value) {"lowBit" = %right_width} -> (%t_left : !pdl.type)
        %cst2_prime = pdl.result 0 of %cst2_prime_op

        %zero = pdl.attribute = 0 : i32
        %cst2_primeprime_op = pdl.operation "comb.extract"(%cst2 : !pdl.value) {"lowBit" = %zero} -> (%t_right : !pdl.type)
        %cst2_primeprime = pdl.result 0 of %cst2_primeprime_op

        %and_x_cst2_prime_op = pdl.operation "comb.and"(%x, %cst2_prime : !pdl.value, !pdl.value) -> (%t_left : !pdl.type)
        %and_x_cst2_prime = pdl.result 0 of %and_x_cst2_prime_op

        %and_cst1_cst2_primeprime_op = pdl.operation "comb.and"(%cst1, %cst2_primeprime : !pdl.value, !pdl.value) -> (%t_right : !pdl.type)
        %and_cst1_cst2_primeprime = pdl.result 0 of %and_cst1_cst2_primeprime_op

        %new_concat_op = pdl.operation "comb.concat"(%and_x_cst2_prime, %and_cst1_cst2_primeprime : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        %new_concat = pdl.result 0 of %new_concat_op

        %new_op = pdl.operation "comb.and"(%a, %b, %c, %new_concat : !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)
        pdl.replace %and_op with %new_op
    }
}

// and(a[0], a[1], ..., a[n]) -> icmp eq(a, -1)
pdl.pattern @AndCommonOperand : benefit(0) {
    %i3 = pdl.type : i3
    %i1 = pdl.type : i1

    %a = pdl.operand : %i3

    %zero = pdl.attribute = 0 : i3
    %one = pdl.attribute = 1 : i3
    %two = pdl.attribute = 2 : i3

    %a0_op = pdl.operation "comb.extract"(%a : !pdl.value) { "lowBit" = %zero } -> (%i1 : !pdl.type)
    %a0 = pdl.result 0 of %a0_op

    %a1_op = pdl.operation "comb.extract"(%a : !pdl.value) { "lowBit" = %one } -> (%i1 : !pdl.type)
    %a1 = pdl.result 0 of %a1_op

    %a2_op = pdl.operation "comb.extract"(%a : !pdl.value) { "lowBit" = %two } -> (%i1 : !pdl.type)
    %a2 = pdl.result 0 of %a2_op

    %and_op = pdl.operation "comb.and"(%a0, %a1, %a2 : !pdl.value, !pdl.value, !pdl.value) -> (%i1 : !pdl.type)

    pdl.rewrite %and_op {
        %minus_one_attr = pdl.apply_native_rewrite "get_minus_one_attr"(%i3 : !pdl.type) : !pdl.attribute
        %minus_one_op = pdl.operation "hw.constant" {"value" = %minus_one_attr} -> (%i1 : !pdl.type)
        %minus_one = pdl.result 0 of %minus_one_op

        %eq_attr = pdl.attribute = 0 : i64

        %icmp_op = pdl.operation "comb.icmp"(%a, %minus_one : !pdl.value, !pdl.value) { "predicate" = %eq_attr } -> (%i1 : !pdl.type)

        pdl.replace %and_op with %icmp_op
    }
}

// CHECK: All patterns are sound
