// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // select(predA, select(predB, x, y), y) => select(and(predA, predB), x, y)
    pdl.pattern @SelectAndCond : benefit(0) {
        %i1 = pdl.type : i1
        %type = pdl.type : !transfer.integer

        %predA = pdl.operand : %i1
        %predB = pdl.operand : %i1
        %x = pdl.operand : %type
        %y = pdl.operand : %type

        %select_op = pdl.operation "arith.select"(%predB, %x, %y : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %select = pdl.result 0 of %select_op

        %select_op2 = pdl.operation "arith.select"(%predA, %select, %y : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %select_op2 {
            %and_op = pdl.operation "arith.andi"(%predA, %predB : !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
            %and = pdl.result 0 of %and_op

            %new_select_op = pdl.operation "arith.select"(%and, %x, %y : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %select_op2 with %new_select_op
        }
    }
}

// CHECK: At least one pattern is unsound
