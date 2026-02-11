// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // subi(addi(x, c0), c1) -> addi(x, c0 - c1)
    pdl.pattern @SubIRHSAddConstant : benefit(0) {
        %type = pdl.type : !transfer.integer<8>

        %c0_attr = pdl.attribute : %type
        %c1_attr = pdl.attribute : %type

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%type : !pdl.type)

        %x = pdl.operand : %type
        %c0 = pdl.result 0 of %c0_op
        %c1 = pdl.result 0 of %c1_op

        %add1_op = pdl.operation "arith.addi"(%x, %c0 : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %add1 = pdl.result 0 of %add1_op

        %add2 = pdl.operation "arith.subi"(%add1, %c1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add2 {
            %res = pdl.apply_native_rewrite "subi"(%c0_attr, %c1_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %add = pdl.operation "arith.addi"(%x, %folded : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %add2 with %add
        }
    }

    // subi(c1, addi(x, c0)) -> subi(c1 - c0, x)
    pdl.pattern @SubILHSAddConstant : benefit(0) {
        %type = pdl.type : !transfer.integer<8>

        %c0_attr = pdl.attribute : %type
        %c1_attr = pdl.attribute : %type

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%type : !pdl.type)

        %x = pdl.operand : %type
        %c0 = pdl.result 0 of %c0_op
        %c1 = pdl.result 0 of %c1_op

        %add1_op = pdl.operation "arith.addi"(%x, %c0 : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %add1 = pdl.result 0 of %add1_op

        %add2 = pdl.operation "arith.subi"(%c1, %add1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add2 {
            %res = pdl.apply_native_rewrite "subi"(%c1_attr, %c0_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %add = pdl.operation "arith.subi"(%folded, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %add2 with %add
        }
    }

    // subi(subi(x, c0), c1) -> subi(x, c0 + c1)
    pdl.pattern @SubIRHSSubConstantRHS : benefit(0) {
        %type = pdl.type : !transfer.integer<8>

        %c0_attr = pdl.attribute : %type
        %c1_attr = pdl.attribute : %type

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%type : !pdl.type)

        %x = pdl.operand : %type
        %c0 = pdl.result 0 of %c0_op
        %c1 = pdl.result 0 of %c1_op

        %sub1_op = pdl.operation "arith.subi"(%x, %c0 : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %sub1 = pdl.result 0 of %sub1_op

        %sub2 = pdl.operation "arith.subi"(%sub1, %c1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %sub2 {
            %res = pdl.apply_native_rewrite "addi"(%c1_attr, %c0_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %sub3 = pdl.operation "arith.subi"(%x, %folded : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %sub2 with %sub3
        }
    }

    // subi(subi(c0, x), c1) -> subi(c0 - c1, x)
    pdl.pattern @SubIRHSSubConstantLHS : benefit(0) {
        %type = pdl.type : !transfer.integer<8>

        %c0_attr = pdl.attribute : %type
        %c1_attr = pdl.attribute : %type

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%type : !pdl.type)

        %x = pdl.operand : %type
        %c0 = pdl.result 0 of %c0_op
        %c1 = pdl.result 0 of %c1_op

        %sub1_op = pdl.operation "arith.subi"(%c0, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %sub1 = pdl.result 0 of %sub1_op

        %sub2 = pdl.operation "arith.subi"(%sub1, %c1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %sub2 {
            %res = pdl.apply_native_rewrite "subi"(%c0_attr, %c1_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %sub3 = pdl.operation "arith.subi"(%folded, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %sub2 with %sub3
        }
    }

    // subi(c1, subi(x, c0)) -> subi(c0 + c1, x)
    pdl.pattern @SubILHSSubConstantRHS : benefit(0) {
        %type = pdl.type : !transfer.integer<8>

        %c0_attr = pdl.attribute : %type
        %c1_attr = pdl.attribute : %type

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%type : !pdl.type)

        %x = pdl.operand : %type
        %c0 = pdl.result 0 of %c0_op
        %c1 = pdl.result 0 of %c1_op

        %sub1_op = pdl.operation "arith.subi"(%x, %c0 : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %sub1 = pdl.result 0 of %sub1_op

        %sub2 = pdl.operation "arith.subi"(%c1, %sub1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %sub2 {
            %res = pdl.apply_native_rewrite "addi"(%c0_attr, %c1_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %sub3 = pdl.operation "arith.subi"(%folded, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %sub2 with %sub3
        }
    }

    // subi(c1, subi(c0, x)) -> addi(x, c1 - c0)
    pdl.pattern @SubILHSSubConstantLHS : benefit(0) {
        %type = pdl.type : !transfer.integer<8>

        %c0_attr = pdl.attribute : %type
        %c1_attr = pdl.attribute : %type

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%type : !pdl.type)

        %x = pdl.operand : %type
        %c0 = pdl.result 0 of %c0_op
        %c1 = pdl.result 0 of %c1_op

        %sub1_op = pdl.operation "arith.subi"(%c0, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %sub1 = pdl.result 0 of %sub1_op

        %sub2 = pdl.operation "arith.subi"(%c1, %sub1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %sub2 {
            %res = pdl.apply_native_rewrite "subi"(%c1_attr, %c0_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %sub3 = pdl.operation "arith.addi"(%x, %folded : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %sub2 with %sub3
        }
    }

    // subi(subi(a, b), a) -> subi(0, b)
    pdl.pattern @SubISubILHSRHSLHS : benefit(0) {
        %type = pdl.type : i32
        %a = pdl.operand : %type
        %b = pdl.operand : %type
        %sub1_op = pdl.operation "arith.subi"(%a, %b : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %sub1 = pdl.result 0 of %sub1_op

        %sub2 = pdl.operation "arith.subi"(%sub1, %a : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %sub2 {
            %zero_attr = pdl.apply_native_rewrite "get_zero_attr"(%type: !pdl.type) : !pdl.attribute
            %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%type : !pdl.type)
            %zero = pdl.result 0 of %zero_op
            %sub3 = pdl.operation "arith.subi"(%zero, %b : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %sub2 with %sub3
        }
    }
}

// CHECK-NOT: UNSOUND
