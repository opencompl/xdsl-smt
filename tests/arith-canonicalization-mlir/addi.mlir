// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // addi(addi(x, c0), c1) -> addi(x, c0 + c1)
    pdl.pattern @add_assoc_add_fold : benefit(0) {
        %type = pdl.type : !transfer.integer

        %c0_attr = pdl.attribute : %type
        %c1_attr = pdl.attribute : %type

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%type : !pdl.type)

        %x = pdl.operand : %type
        %c0 = pdl.result 0 of %c0_op
        %c1 = pdl.result 0 of %c1_op

        %add1_op = pdl.operation "arith.addi"(%x, %c0 : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %add1 = pdl.result 0 of %add1_op

        %add2 = pdl.operation "arith.addi"(%add1, %c1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add2 {
            %res = pdl.apply_native_rewrite "addi"(%c0_attr, %c1_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %add = pdl.operation "arith.addi"(%x, %folded : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %add2 with %add
        }
    }

    // addi(subi(x, c0), c1) -> addi(x, c0 - c1)
    pdl.pattern @add_assoc_sub_fold : benefit(0) {
        %type = pdl.type : !transfer.integer

        %c0_attr = pdl.attribute : %type
        %c1_attr = pdl.attribute : %type

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%type : !pdl.type)

        %x = pdl.operand : %type
        %c0 = pdl.result 0 of %c0_op
        %c1 = pdl.result 0 of %c1_op

        %sub1_op = pdl.operation "arith.subi"(%x, %c0 : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %sub1 = pdl.result 0 of %sub1_op

        %add2 = pdl.operation "arith.addi"(%sub1, %c1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add2 {
            %res = pdl.apply_native_rewrite "subi"(%c1_attr, %c0_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %add = pdl.operation "arith.addi"(%x, %folded : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %add2 with %add
        }
    }

    // addi(subi(c0, x), c1) -> addi(c0 + c1, x)
    pdl.pattern @add_assoc_sub_fold2 : benefit(0) {
        %type = pdl.type : !transfer.integer

        %c0_attr = pdl.attribute : %type
        %c1_attr = pdl.attribute : %type

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%type : !pdl.type)

        %x = pdl.operand : %type
        %c0 = pdl.result 0 of %c0_op
        %c1 = pdl.result 0 of %c1_op

        %sub1_op = pdl.operation "arith.subi"(%c0, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %sub1 = pdl.result 0 of %sub1_op

        %add2 = pdl.operation "arith.addi"(%sub1, %c1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add2 {
            %res = pdl.apply_native_rewrite "addi"(%c0_attr, %c1_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %sub = pdl.operation "arith.subi"(%folded, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %add2 with %sub
        }
    }
}

// CHECK-NOT: UNSOUND
