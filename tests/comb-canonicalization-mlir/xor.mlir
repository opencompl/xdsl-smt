// RUN: verify-pdl "%s" -opt | filecheck "%s"

// extracts only of xor(...) -> xor(extract()...)

// xor(x) -> x
pdl.pattern @XorSingle : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t

    %xor_op = pdl.operation "comb.xor"(%x : !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %xor_op {
        pdl.replace %xor_op with (%x : !pdl.value)
    }
}

// xor(x, x) -> 0
pdl.pattern @XorSame : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t

    %xor_op = pdl.operation "comb.xor"(%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %xor_op {
        %zero_attr = pdl.apply_native_rewrite "get_zero_attr"(%t : !pdl.type) : !pdl.value
        %zero_op = pdl.operation "hw.constant" {"value" = %zero_attr} -> (%t : !pdl.type)

        pdl.replace %xor_op with %zero_op
    }
}

// xor(a, b, x, x) -> xor(a, b)
pdl.pattern @XorSameExtended : benefit(0) {
    %t = pdl.type : !transfer.integer
    %a = pdl.operand : %t
    %b = pdl.operand : %t
    %x = pdl.operand : %t

    %xor_op = pdl.operation "comb.xor"(%a, %b, %x, %x : !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %xor_op {
        %new_xor_op = pdl.operation "comb.xor"(%a, %b : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        pdl.replace %xor_op with %new_xor_op
    }
}

// xor(x, 0) -> x
pdl.pattern @XorZero : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t

    %zero_attr = pdl.attribute : %t
    pdl.apply_native_constraint "is_zero"(%zero_attr : !pdl.attribute)

    %zero_op = pdl.operation "hw.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
    %zero = pdl.result 0 of %zero_op

    %xor_op = pdl.operation "comb.xor"(%x, %zero : !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %xor_op {
        pdl.replace %xor_op with (%x : !pdl.value)
    }
}

// xor(a, b, 0) -> xor(a, b)
pdl.pattern @XorZeroExtended : benefit(0) {
    %t = pdl.type : !transfer.integer
    %a = pdl.operand : %t
    %b = pdl.operand : %t

    %zero_attr = pdl.attribute : %t
    pdl.apply_native_constraint "is_zero"(%zero_attr : !pdl.attribute)

    %zero_op = pdl.operation "hw.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
    %zero = pdl.result 0 of %zero_op

    %xor_op = pdl.operation "comb.xor"(%a, %b, %zero : !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %xor_op {
        %new_xor_op = pdl.operation "comb.xor"(%a, %b : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        pdl.replace %xor_op with %new_xor_op
    }
}

// xor(x, cst1, cst2) -> xor(x, cst1 ^ cst2)
pdl.pattern @OrMinusOne : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t

    %cst1_attr = pdl.attribute : %t
    %cst1_op = pdl.operation "hw.constant" {"value" = %cst1_attr} -> (%t: !pdl.type)
    %cst1 = pdl.result 0 of %cst1_op

    %cst2_attr = pdl.attribute : %t
    %cst2_op = pdl.operation "hw.constant" {"value" = %cst2_attr} -> (%t: !pdl.type)
    %cst2 = pdl.result 0 of %cst2_op

    %xor_op = pdl.operation "comb.xor"(%x, %cst1, %cst2 : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %xor_op {
        %merged_cst = pdl.apply_native_rewrite "xori"(%cst1_attr, %cst2_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
        %cst_op = pdl.operation "hw.constant" {"value" = %merged_cst} -> (%t: !pdl.type)
        %cst = pdl.result 0 of %cst_op
        %new_op = pdl.operation "comb.xor"(%x, %cst: !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %xor_op with %new_op
    }
}

// xor(concat(x, cst1), a, b, c, cst2)
//    ==> xor(a, b, c, concat(xor(x,cst2'), xor(cst1,cst2'')).
pdl.pattern @XOrConcatCst : benefit(0) {
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

    %xor_op = pdl.operation "comb.xor"(%concat, %a, %b, %c, %cst2 : !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %xor_op {
        %right_width = pdl.apply_native_rewrite "get_width"(%t_right, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
        %cst2_prime_op = pdl.operation "comb.extract"(%cst2 : !pdl.value) {"low_bit" = %right_width} -> (%t_left : !pdl.type)
        %cst2_prime = pdl.result 0 of %cst2_prime_op

        %zero = pdl.attribute = 0 : i32
        %cst2_primeprime_op = pdl.operation "comb.extract"(%cst2 : !pdl.value) {"low_bit" = %zero} -> (%t_right : !pdl.type)
        %cst2_primeprime = pdl.result 0 of %cst2_primeprime_op

        %and_x_cst2_prime_op = pdl.operation "comb.xor"(%x, %cst2_prime : !pdl.value, !pdl.value) -> (%t_left : !pdl.type)
        %and_x_cst2_prime = pdl.result 0 of %and_x_cst2_prime_op

        %and_cst1_cst2_primeprime_op = pdl.operation "comb.xor"(%cst1, %cst2_primeprime : !pdl.value, !pdl.value) -> (%t_right : !pdl.type)
        %and_cst1_cst2_primeprime = pdl.result 0 of %and_cst1_cst2_primeprime_op

        %new_concat_op = pdl.operation "comb.concat"(%and_x_cst2_prime, %and_cst1_cst2_primeprime : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        %new_concat = pdl.result 0 of %new_concat_op

        %new_op = pdl.operation "comb.xor"(%a, %b, %c, %new_concat : !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)
        pdl.replace %xor_op with %new_op
    }
}

// xor(icmp, a, b, 1) -> xor(icmp, a, b)
pdl.pattern @XorIcmpTrue : benefit(0) {
    %i1 = pdl.type : i1
    %i64 = pdl.type : i64

    %predicate = pdl.attribute : %i64
    pdl.apply_native_constraint "is_comb_icmp_predicate"(%predicate : !pdl.attribute)
    %x = pdl.operand : %i1
    %y = pdl.operand : %i1

    %icmp_op = pdl.operation "comb.icmp"(%x, %y : !pdl.value, !pdl.value) {"predicate" = %predicate} -> (%i1 : !pdl.type)
    %icmp = pdl.result 0 of %icmp_op

    %a = pdl.operand : %i1
    %b = pdl.operand : %i1

    %one_attr = pdl.attribute : %i1
    pdl.apply_native_constraint "is_one"(%one_attr : !pdl.attribute)
    %one_op = pdl.operation "hw.constant" {"value" = %one_attr} -> (%i1 : !pdl.type)
    %one = pdl.result 0 of %one_op

    %xor_op = pdl.operation "comb.xor"(%icmp, %a, %b, %one : !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%i1 : !pdl.type)

    pdl.rewrite %xor_op {
        %negated_predicate = pdl.apply_native_rewrite "invert_comb_icmp_predicate"(%predicate : !pdl.attribute) : !pdl.attribute
        %new_icmp_op = pdl.operation "comb.icmp"(%x, %y : !pdl.value, !pdl.value) {"predicate" = %negated_predicate} -> (%i1 : !pdl.type)
        %new_icmp = pdl.result 0 of %new_icmp_op

        %new_op = pdl.operation "comb.xor"(%new_icmp, %a, %b : !pdl.value, !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
        pdl.replace %xor_op with %new_op
    }
}

// xor(x, xor(val1, val2)) -> xor(x, val1, val2) -- flatten
pdl.pattern @XOrFlatten : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t
    %val1 = pdl.operand : %t
    %val2 = pdl.operand : %t

    %inner_or_op = pdl.operation "comb.xor"(%val1, %val2 : !pdl.value, !pdl.value) -> (%t: !pdl.type)
    %inner_and = pdl.result 0 of %inner_or_op

    %or_op = pdl.operation "comb.xor"(%x, %inner_and : !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %or_op {
        %new_op = pdl.operation "comb.xor"(%x, %val1, %val2 : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %or_op with %new_op
    }
}

// xor(a[0], a[1], ..., a[n]) -> parity(a)
pdl.pattern @OrCommonOperand : benefit(0) {
    %i3 = pdl.type : i3
    %i1 = pdl.type : i1

    %a = pdl.operand : %i3

    %zero = pdl.attribute = 0 : i3
    %one = pdl.attribute = 1 : i3
    %two = pdl.attribute = 2 : i3

    %a0_op = pdl.operation "comb.extract"(%a : !pdl.value) { "low_bit" = %zero } -> (%i1 : !pdl.type)
    %a0 = pdl.result 0 of %a0_op

    %a1_op = pdl.operation "comb.extract"(%a : !pdl.value) { "low_bit" = %one } -> (%i1 : !pdl.type)
    %a1 = pdl.result 0 of %a1_op

    %a2_op = pdl.operation "comb.extract"(%a : !pdl.value) { "low_bit" = %two } -> (%i1 : !pdl.type)
    %a2 = pdl.result 0 of %a2_op

    %or_op = pdl.operation "comb.xor"(%a0, %a1, %a2 : !pdl.value, !pdl.value, !pdl.value) -> (%i1 : !pdl.type)

    pdl.rewrite %or_op {
        %parity_op = pdl.operation "comb.parity"(%a : !pdl.value) -> (%i1 : !pdl.type)

        pdl.replace %or_op with %parity_op
    }
}

// CHECK: All patterns are sound