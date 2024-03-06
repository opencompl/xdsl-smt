// RUN: verify-pdl "%s" -opt | filecheck "%s"

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

// xor(xor(x, 1), 1) -> x
pdl.pattern @XorNegNeg : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t

    %one_attr = pdl.attribute : %t
    pdl.apply_native_constraint "is_one"(%one_attr : !pdl.attribute)

    %one_attr2 = pdl.attribute : %t
    pdl.apply_native_constraint "is_one"(%one_attr2 : !pdl.attribute)


    %one_op = pdl.operation "hw.constant" {"value" = %one_attr} -> (%t : !pdl.type)
    %one = pdl.result 0 of %one_op

    %one_op2 = pdl.operation "hw.constant" {"value" = %one_attr2} -> (%t : !pdl.type)
    %one2 = pdl.result 0 of %one_op2

    %xor_op1 = pdl.operation "comb.xor"(%x, %one : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    %xor1 = pdl.result 0 of %xor_op1

    %xor_op2 = pdl.operation "comb.xor"(%xor1, %one2 : !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %xor_op2 {
        pdl.replace %xor_op2 with (%x : !pdl.value)
    }
}
