// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // select(predA, x, select(predB, x, y)) => select(or(predA, predB), x, y)
    pdl.pattern @SelectOrCond : benefit(0) {
        %i1 = pdl.type : i1
        %type = pdl.type : !transfer.integer<8>

        %predA = pdl.operand : %i1
        %predB = pdl.operand : %i1
        %x = pdl.operand : %type
        %y = pdl.operand : %type

        %select_op = pdl.operation "arith.select"(%predB, %x, %y : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %select = pdl.result 0 of %select_op

        %select_op2 = pdl.operation "arith.select"(%predA, %x, %select : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %select_op2 {
            %or_op = pdl.operation "arith.ori"(%predA, %predB : !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
            %or = pdl.result 0 of %or_op

            %new_select_op = pdl.operation "arith.select"(%or, %x, %y : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %select_op2 with %new_select_op
        }
    }
}

// CHECK: At least one pattern is unsound
