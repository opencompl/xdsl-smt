// RUN: verify-pdl "%s" -opt | filecheck "%s"

// Missing:
// or(..., concat(x, cst1), concat(cst2, y) ==> or(..., concat(x, cst3, y)), when x and y don't overlap.
// extracts only of or(...) -> or(extract()...)


// or(x, or(val1, val2)) -> or(x, val1, val2) -- flatten
pdl.pattern @OrFlatten : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t
    %val1 = pdl.operand : %t
    %val2 = pdl.operand : %t

    %inner_or_op = pdl.operation "comb.or"(%val1, %val2 : !pdl.value, !pdl.value) -> (%t: !pdl.type)
    %inner_and = pdl.result 0 of %inner_or_op

    %or_op = pdl.operation "comb.or"(%x, %inner_and : !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %or_op {
        %new_op = pdl.operation "comb.or"(%x, %val1, %val2 : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %or_op with %new_op
    }
}

// or(x, val1, x) -> or(x, val1) -- idempotent
pdl.pattern @OrIdempotent : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t
    %val1 = pdl.operand : %t

    %or_op = pdl.operation "comb.or"(%x, %val1, %x : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %or_op {
        %new_op = pdl.operation "comb.or"(%x, %val1: !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %or_op with %new_op
    }
}

// or(x, 0, y) -> or(x, y)
pdl.pattern @OrMinusOne : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t
    %y = pdl.operand : %t

    %cst_attr = pdl.attribute : %t
    pdl.apply_native_constraint "is_zero"(%cst_attr : !pdl.attribute)

    %cst_op = pdl.operation "hw.constant" {"value" = %cst_attr} -> (%t: !pdl.type)
    %cst = pdl.result 0 of %cst_op

    %or_op = pdl.operation "comb.or"(%x, %cst, %y : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %or_op {
        %new_op = pdl.operation "comb.or"(%x, %y: !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %or_op with %new_op
    }
}

// or(x, 0) -> x
pdl.pattern @OrMinusOne : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t

    %cst_attr = pdl.attribute : %t
    pdl.apply_native_constraint "is_zero"(%cst_attr : !pdl.attribute)

    %cst_op = pdl.operation "hw.constant" {"value" = %cst_attr} -> (%t: !pdl.type)
    %cst = pdl.result 0 of %cst_op

    %or_op = pdl.operation "comb.or"(%x, %cst : !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %or_op {
        pdl.replace %or_op with (%x : !pdl.value)
    }
}

// or(x, cst1, cst2) -> or(x, cst1 | cst2)
pdl.pattern @OrConstantFolding : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t

    %cst1_attr = pdl.attribute : %t
    %cst1_op = pdl.operation "hw.constant" {"value" = %cst1_attr} -> (%t: !pdl.type)
    %cst1 = pdl.result 0 of %cst1_op

    %cst2_attr = pdl.attribute : %t
    %cst2_op = pdl.operation "hw.constant" {"value" = %cst2_attr} -> (%t: !pdl.type)
    %cst2 = pdl.result 0 of %cst2_op

    %or_op = pdl.operation "comb.or"(%x, %cst1, %cst2 : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %or_op {
        %merged_cst = pdl.apply_native_rewrite "ori"(%cst1_attr, %cst2_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
        %cst_op = pdl.operation "hw.constant" {"value" = %merged_cst} -> (%t: !pdl.type)
        %cst = pdl.result 0 of %cst_op
        %new_op = pdl.operation "comb.or"(%x, %cst: !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %or_op with %new_op
    }
}

// or(concat(x, cst1), a, b, c, cst2)
//    ==> or(a, b, c, concat(or(x,cst2'), or(cst1,cst2'')).
pdl.pattern @OrConcatCst : benefit(0) {
    %t = pdl.type : !transfer.integer
    %t_left = pdl.type : !transfer.integer

    pdl.apply_native_constraint "is_greater_integer_type"(%t, %t_left : !pdl.type, !pdl.type)
    %t_right = pdl.apply_native_rewrite "integer_type_sub_width"(%t, %t_left : !pdl.type, !pdl.type) : !pdl.type

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

    %or_op = pdl.operation "comb.or"(%concat, %a, %b, %c, %cst2 : !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %or_op {
        %i32 = pdl.type : i32
        %right_width = pdl.apply_native_rewrite "get_width"(%t_right, %i32 : !pdl.type, !pdl.type) : !pdl.attribute
        %cst2_prime_op = pdl.operation "comb.extract"(%cst2 : !pdl.value) {"low_bit" = %right_width} -> (%t_left : !pdl.type)
        %cst2_prime = pdl.result 0 of %cst2_prime_op

        %zero = pdl.attribute = 0 : i32
        %cst2_primeprime_op = pdl.operation "comb.extract"(%cst2 : !pdl.value) {"low_bit" = %zero} -> (%t_right : !pdl.type)
        %cst2_primeprime = pdl.result 0 of %cst2_primeprime_op

        %and_x_cst2_prime_op = pdl.operation "comb.or"(%x, %cst2_prime : !pdl.value, !pdl.value) -> (%t_left : !pdl.type)
        %and_x_cst2_prime = pdl.result 0 of %and_x_cst2_prime_op

        %and_cst1_cst2_primeprime_op = pdl.operation "comb.or"(%cst1, %cst2_primeprime : !pdl.value, !pdl.value) -> (%t_right : !pdl.type)
        %and_cst1_cst2_primeprime = pdl.result 0 of %and_cst1_cst2_primeprime_op

        %new_concat_op = pdl.operation "comb.concat"(%and_x_cst2_prime, %and_cst1_cst2_primeprime : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        %new_concat = pdl.result 0 of %new_concat_op

        %new_op = pdl.operation "comb.or"(%a, %b, %c, %new_concat : !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)
        pdl.replace %or_op with %new_op
    }
}

// or(a[0], a[1], ..., a[n]) -> icmp ne(a, 0)
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

    %or_op = pdl.operation "comb.or"(%a0, %a1, %a2 : !pdl.value, !pdl.value, !pdl.value) -> (%i1 : !pdl.type)

    pdl.rewrite %or_op {
        %zero_attr = pdl.apply_native_rewrite "get_zero_attr"(%i3 : !pdl.type) : !pdl.attribute
        %zero_op = pdl.operation "hw.constant" {"value" = %zero_attr} -> (%i1 : !pdl.type)
        %zero_res = pdl.result 0 of %zero_op

        %ne_attr = pdl.attribute = 1 : i64

        %icmp_op = pdl.operation "comb.icmp"(%a, %zero_res : !pdl.value, !pdl.value) { "predicate" = %ne_attr } -> (%i1 : !pdl.type)

        pdl.replace %or_op with %icmp_op
    }
}

// or(mux(c_1, a, 0), mux(c_2, a, 0), ..., mux(c_n, a, 0)) -> mux(or(c_1, c_2, .., c_n), a, 0)
// This is only implemented with 3 values, as we do not have variadics yet
pdl.pattern @OrMuxSwap : benefit(0) {
    %t = pdl.type : !transfer.integer
    %i1 = pdl.type : i1

    %a = pdl.operand : %t

    %zero_attr = pdl.apply_native_rewrite "get_zero_attr"(%t : !pdl.type) : !pdl.attribute
    %zero_op = pdl.operation "hw.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
    %zero = pdl.result 0 of %zero_op

    %c0 = pdl.operand : %i1
    %c1 = pdl.operand : %i1
    %c2 = pdl.operand : %i1

    %mux1_op = pdl.operation "comb.mux"(%c0, %a, %zero : !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)
    %mux1 = pdl.result 0 of %mux1_op

    %mux2_op = pdl.operation "comb.mux"(%c1, %a, %zero : !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)
    %mux2 = pdl.result 0 of %mux2_op

    %mux3_op = pdl.operation "comb.mux"(%c2, %a, %zero : !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)
    %mux3 = pdl.result 0 of %mux3_op

    %or_op = pdl.operation "comb.or"(%mux1, %mux2, %mux3 : !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %or_op {
        %new_or_op = pdl.operation "comb.or"(%c0, %c1, %c2 : !pdl.value, !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
        %new_or = pdl.result 0 of %new_or_op

        %mux_op = pdl.operation "comb.mux"(%new_or, %a, %zero : !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)

        pdl.replace %or_op with %mux_op
    }
}

// CHECK: All patterns are sound
