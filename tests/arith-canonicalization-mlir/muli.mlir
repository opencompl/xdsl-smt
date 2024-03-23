// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // muli(muli(x, c0), c1) -> muli(x, c0 * c1)
    pdl.pattern @MulIMulIConstant : benefit(0) {
        %type = pdl.type : !transfer.integer

        %c0_attr = pdl.attribute : %type
        %c1_attr = pdl.attribute : %type

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)
        %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%type : !pdl.type)

        %x = pdl.operand : %type
        %c0 = pdl.result 0 of %c0_op
        %c1 = pdl.result 0 of %c1_op

        %add1_op = pdl.operation "arith.muli"(%x, %c0 : !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %add1 = pdl.result 0 of %add1_op

        %add2 = pdl.operation "arith.muli"(%add1, %c1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add2 {
            %res = pdl.apply_native_rewrite "muli"(%c0_attr, %c1_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %folded_op = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            %folded = pdl.result 0 of %folded_op
            %add = pdl.operation "arith.muli"(%x, %folded : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %add2 with %add
        }
    }
}

// CHECK-NOT: UNSOUND
