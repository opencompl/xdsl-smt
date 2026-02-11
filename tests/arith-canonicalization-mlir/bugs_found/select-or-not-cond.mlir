// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // select(predA, x, select(predB, y, x)) => select(or(predA, not(predB)), x, y)
    pdl.pattern @SelectOrNotCond : benefit(0) {
        %i1 = pdl.type : i1
        %type = pdl.type : !transfer.integer<8>

        %predA = pdl.operand : %i1
        %predB = pdl.operand : %i1
        %x = pdl.operand : %type
        %y = pdl.operand : %type

        %select_op = pdl.operation "arith.select"(%predB, %y, %x : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %select = pdl.result 0 of %select_op

        %select_op2 = pdl.operation "arith.select"(%predA, %x, %select : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %select_op2 {
            %one_attr = pdl.attribute = 1 : i1
            %one_op = pdl.operation "arith.constant" {"value" = %one_attr} -> (%i1 : !pdl.type)
            %one = pdl.result 0 of %one_op

            %not_predB_op = pdl.operation "arith.xori"(%predB, %one : !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
            %not_predB = pdl.result 0 of %not_predB_op

            %and_op = pdl.operation "arith.ori"(%predA, %not_predB : !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
            %and = pdl.result 0 of %and_op

            %new_select_op = pdl.operation "arith.select"(%and, %x, %y : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %select_op2 with %new_select_op
        }
    }
}

// CHECK: At least one pattern is unsound
