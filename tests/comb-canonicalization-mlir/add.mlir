// RUN: verify-pdl "%s" -opt | filecheck "%s"

// Missing:
// extracts only of add(...) -> add(extract()...)

// add(a, b, 0) -> add(a, b)
pdl.pattern @AddZero : benefit(0) {
    %t = pdl.type : !transfer.integer<8>
    %a = pdl.operand : %t
    %b = pdl.operand : %t

    %zero_attr = pdl.attribute : %t
    pdl.apply_native_constraint "is_zero"(%zero_attr : !pdl.attribute)
    %zero_op = pdl.operation "hw.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
    %zero = pdl.result 0 of %zero_op

    %add_op = pdl.operation "comb.add"(%a, %b, %zero : !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %add_op {
        %new_add = pdl.operation "comb.add"(%a, %b : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        pdl.replace %add_op with %new_add
    }
}

// add(x, cst1, cst2) -> add(x, cst1 + cst2)
pdl.pattern @AddConstantFolding : benefit(0) {
    %t = pdl.type : !transfer.integer<8>
    %x = pdl.operand : %t

    %cst1_attr = pdl.attribute : %t
    %cst1_op = pdl.operation "hw.constant" {"value" = %cst1_attr} -> (%t: !pdl.type)
    %cst1 = pdl.result 0 of %cst1_op

    %cst2_attr = pdl.attribute : %t
    %cst2_op = pdl.operation "hw.constant" {"value" = %cst2_attr} -> (%t: !pdl.type)
    %cst2 = pdl.result 0 of %cst2_op

    %or_op = pdl.operation "comb.add"(%x, %cst1, %cst2 : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %or_op {
        %merged_cst = pdl.apply_native_rewrite "addi"(%cst1_attr, %cst2_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
        %cst_op = pdl.operation "hw.constant" {"value" = %merged_cst} -> (%t: !pdl.type)
        %cst = pdl.result 0 of %cst_op
        %new_op = pdl.operation "comb.add"(%x, %cst: !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %or_op with %new_op
    }
}

// add(..., x, x) -> add(..., shl(x, 1))
pdl.pattern @AddTwice : benefit(0) {
    %t = pdl.type : !transfer.integer<8>
    %a = pdl.operand : %t
    %x = pdl.operand : %t

    %add_op = pdl.operation "comb.add"(%a, %x, %x : !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %add_op {
        %one_attr = pdl.apply_native_rewrite "get_one_attr"(%t : !pdl.type) : !pdl.attribute
        %one_op = pdl.operation "hw.constant" {"value" = %one_attr} -> (%t : !pdl.type)
        %one = pdl.result 0 of %one_op

        %shl_op = pdl.operation "comb.shl"(%x, %one : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        %shl = pdl.result 0 of %shl_op

        %new_op = pdl.operation "comb.add"(%a, %shl : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        pdl.replace %add_op with %new_op
    }
}

// add(..., x, shl(x, c)) -> add(..., mul(x, (1 << c) + 1))
pdl.pattern @AddShift : benefit(0) {
    %t = pdl.type : !transfer.integer<8>
    %a = pdl.operand : %t
    %x = pdl.operand : %t

    %c_attr = pdl.attribute : %t
    %c_op = pdl.operation "hw.constant" {"value" = %c_attr} -> (%t : !pdl.type)
    %c = pdl.result 0 of %c_op

    %shl_op = pdl.operation "comb.shl"(%x, %c : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    %shl = pdl.result 0 of %shl_op

    %add_op = pdl.operation "comb.add"(%a, %x, %shl : !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %add_op {
        %one_attr = pdl.apply_native_rewrite "get_one_attr"(%t : !pdl.type) : !pdl.attribute
        %shl_cst_attr = pdl.apply_native_rewrite "shl"(%one_attr, %c_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
        %mul_cst_attr = pdl.apply_native_rewrite "addi"(%one_attr, %shl_cst_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
        %mul_cst_op = pdl.operation "hw.constant" {"value" = %mul_cst_attr} -> (%t : !pdl.type)
        %mul_cst = pdl.result 0 of %mul_cst_op

        %mul_op = pdl.operation "comb.mul"(%x, %mul_cst : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        %mul = pdl.result 0 of %mul_op

        %new_op = pdl.operation "comb.add"(%a, %mul : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        pdl.replace %add_op with %new_op
    }
}

// add(..., x, mul(x, c)) -> add(..., mul(x, c + 1))
pdl.pattern @AddMul : benefit(0) {
    %t = pdl.type : !transfer.integer<8>
    %a = pdl.operand : %t
    %x = pdl.operand : %t

    %c_attr = pdl.attribute : %t
    %c_op = pdl.operation "hw.constant" {"value" = %c_attr} -> (%t : !pdl.type)
    %c = pdl.result 0 of %c_op

    %mul_op = pdl.operation "comb.mul"(%x, %c : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    %mul = pdl.result 0 of %mul_op

    %add_op = pdl.operation "comb.add"(%a, %x, %mul : !pdl.value, !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %add_op {
        %one_attr = pdl.apply_native_rewrite "get_one_attr"(%t : !pdl.type) : !pdl.attribute
        %mul_cst_attr = pdl.apply_native_rewrite "addi"(%one_attr, %c_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
        %mul_cst_op = pdl.operation "hw.constant" {"value" = %mul_cst_attr} -> (%t : !pdl.type)
        %mul_cst = pdl.result 0 of %mul_cst_op

        %new_mul_op = pdl.operation "comb.mul"(%x, %mul_cst : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        %new_mul = pdl.result 0 of %new_mul_op

        %new_op = pdl.operation "comb.add"(%a, %new_mul : !pdl.value, !pdl.value) -> (%t : !pdl.type)
        pdl.replace %add_op with %new_op
    }
}

// add(x, add(val1, val2)) -> add(x, val1, val2) -- flatten
pdl.pattern @AddFlatten : benefit(0) {
    %t = pdl.type : !transfer.integer<8>
    %x = pdl.operand : %t
    %val1 = pdl.operand : %t
    %val2 = pdl.operand : %t

    %inner_or_op = pdl.operation "comb.add"(%val1, %val2 : !pdl.value, !pdl.value) -> (%t: !pdl.type)
    %inner_and = pdl.result 0 of %inner_or_op

    %or_op = pdl.operation "comb.add"(%x, %inner_and : !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %or_op {
        %new_op = pdl.operation "comb.add"(%x, %val1, %val2 : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %or_op with %new_op
    }
}

// add(add(x, c1), c2) -> add(x, c1 + c2)
pdl.pattern @AddFolding : benefit(0) {
    %t = pdl.type : !transfer.integer<8>
    %x = pdl.operand : %t

    %cst1_attr = pdl.attribute : %t
    %cst1_op = pdl.operation "hw.constant" {"value" = %cst1_attr} -> (%t: !pdl.type)
    %cst1 = pdl.result 0 of %cst1_op

    %cst2_attr = pdl.attribute : %t
    %cst2_op = pdl.operation "hw.constant" {"value" = %cst2_attr} -> (%t: !pdl.type)
    %cst2 = pdl.result 0 of %cst2_op

    %add_op = pdl.operation "comb.add"(%x, %cst1, %cst2 : !pdl.value, !pdl.value, !pdl.value) -> (%t: !pdl.type)

    pdl.rewrite %add_op {
        %merged_cst = pdl.apply_native_rewrite "addi"(%cst1_attr, %cst2_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
        %cst_op = pdl.operation "hw.constant" {"value" = %merged_cst} -> (%t: !pdl.type)
        %cst = pdl.result 0 of %cst_op
        %new_op = pdl.operation "comb.add"(%x, %cst: !pdl.value, !pdl.value) -> (%t: !pdl.type)
        pdl.replace %add_op with %new_op
    }
}

// CHECK: All patterns are sound
