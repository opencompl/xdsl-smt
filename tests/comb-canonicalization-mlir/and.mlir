// RUN: verify-pdl "%s" -opt | filecheck "%s"

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

// CHECK: All patterns are sound
