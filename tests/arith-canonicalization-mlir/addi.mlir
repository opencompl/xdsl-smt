// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // addi(addi(x, c0), c1) -> addi(x, c0 + c1)
    pdl.pattern @AddIAddConstant : benefit(0) {
        %type = pdl.type : !transfer.integer<8>

        %c0_attr = pdl.attribute : %type
        %c1_attr = pdl.attribute : %type

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%type : !pdl.type)

        %x = pdl.operand : %type
        %c0 = pdl.result 0 of %c0_op
        %c1 = pdl.result 0 of %c1_op

        %ovf1 = pdl.attribute attributes {base_type = "arith.overflow"}
        %add1_op = pdl.operation "arith.addi"(%x, %c0 : !pdl.value, !pdl.value) {"overflowFlags" = %ovf1} -> (%type : !pdl.type)
        %add1 = pdl.result 0 of %add1_op

        %ovf2 = pdl.attribute attributes {base_type = "arith.overflow"}
        %add2 = pdl.operation "arith.addi"(%add1, %c1 : !pdl.value, !pdl.value) {"overflowFlags" = %ovf2} -> (%type : !pdl.type)

        pdl.rewrite %add2 {
            %res = pdl.apply_native_rewrite "addi"(%c0_attr, %c1_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %ovf = pdl.apply_native_rewrite "merge_overflow"(%ovf1, %ovf2 : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %add = pdl.operation "arith.addi"(%x, %folded : !pdl.value, !pdl.value) {"overflowFlags" = %ovf} -> (%type : !pdl.type)
            pdl.replace %add2 with %add
        }
    }

    // addi(subi(x, c0), c1) -> addi(x, c0 - c1)
    pdl.pattern @AddISubConstantRHS : benefit(0) {
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
    pdl.pattern @AddISubConstantLHS : benefit(0) {
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

        %add2 = pdl.operation "arith.addi"(%sub1, %c1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add2 {
            %res = pdl.apply_native_rewrite "addi"(%c0_attr, %c1_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %sub = pdl.operation "arith.subi"(%folded, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %add2 with %sub
        }
    }

    // addi(x, muli(y, -1)) -> subi(x, y)
    pdl.pattern @AddIMulNegativeOneRhs : benefit(0) {
        %type = pdl.type : !transfer.integer<8>

        %x = pdl.operand : %type
        %y = pdl.operand : %type
        %m1_attr = pdl.attribute : %type

        pdl.apply_native_constraint "is_minus_one"(%m1_attr : !pdl.attribute)

        %m1_op = pdl.operation "arith.constant" {"value" = %m1_attr} -> (%type : !pdl.type)
        %m1 = pdl.result 0 of %m1_op

        %mul_op = pdl.operation "arith.muli"(%y, %m1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %mul = pdl.result 0 of %mul_op

        %add = pdl.operation "arith.addi"(%x, %mul : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add {
            %sub = pdl.operation "arith.subi"(%x, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %add with %sub
        }
    }

    // addi(muli(x, -1), y) -> subi(y, x)
    pdl.pattern @AddIMulNegativeOneLhs : benefit(0) {
        %type = pdl.type : !transfer.integer<8>

        %x = pdl.operand : %type
        %y = pdl.operand : %type
        %m1_attr = pdl.attribute : %type

        pdl.apply_native_constraint "is_minus_one"(%m1_attr : !pdl.attribute)

        %m1_op = pdl.operation "arith.constant" {"value" = %m1_attr} -> (%type : !pdl.type)
        %m1 = pdl.result 0 of %m1_op

        %mul_op = pdl.operation "arith.muli"(%x, %m1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %mul = pdl.result 0 of %mul_op

        %add = pdl.operation "arith.addi"(%mul, %y : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add {
            %sub = pdl.operation "arith.subi"(%y, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %add with %sub
        }
    }
}

// CHECK-NOT: UNSOUND
