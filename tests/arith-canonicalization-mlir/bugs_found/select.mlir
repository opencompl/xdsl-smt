// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // Transforms a select of a boolean to arithmetic operations
    //
    //  arith.select %pred, %x, %y : i1
    //
    //  becomes
    //
    //  and(%pred, %x) or and(not(%pred), %y) where not(x) = xor(x, 1)
    pdl.pattern @SelectOrNotCond : benefit(0) {
        %i1 = pdl.type : i1

        %pred = pdl.operand : %i1
        %x = pdl.operand : %i1
        %y = pdl.operand : %i1

        %select_op = pdl.operation "arith.select"(%pred, %x, %y : !pdl.value, !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
        %select = pdl.result 0 of %select_op

        pdl.rewrite %select_op {
            %one_attr = pdl.attribute = 1 : i1
            %one_op = pdl.operation "arith.constant" {"value" = %one_attr} -> (%i1 : !pdl.type)
            %one = pdl.result 0 of %one_op

            %not_pred_op = pdl.operation "arith.xori"(%pred, %one : !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
            %not_pred = pdl.result 0 of %not_pred_op

            %x_choice_op = pdl.operation "arith.andi"(%pred, %x : !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
            %x_choice = pdl.result 0 of %x_choice_op

            %y_choice_op = pdl.operation "arith.andi"(%not_pred, %y : !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
            %y_choice = pdl.result 0 of %y_choice_op

            %res_op = pdl.operation "arith.ori"(%x_choice, %y_choice : !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
            pdl.replace %select_op with %res_op
        }
    }
}

// CHECK: At least one pattern is unsound
