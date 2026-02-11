// RUN: verify-pdl "%s" -opt | filecheck "%s"

builtin.module {
    // select(not(pred), a, b) => select(pred, b, a)
    pdl.pattern @SelectNotCond : benefit(0) {
        %i1 = pdl.type : i1
        %type = pdl.type : !transfer.integer<8>

        %pred = pdl.operand : %i1
        %a = pdl.operand : %type
        %b = pdl.operand : %type

        %one_attr = pdl.attribute = 1 : i1
        %one_op = pdl.operation "arith.constant" {"value" = %one_attr} -> (%i1 : !pdl.type)
        %one = pdl.result 0 of %one_op

        %not_pred_op = pdl.operation "arith.xori"(%pred, %one : !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
        %not_pred = pdl.result 0 of %not_pred_op

        %select_op = pdl.operation "arith.select"(%not_pred, %a, %b : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %select_op {
            %new_select_op = pdl.operation "arith.select"(%pred, %b, %a : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %select_op with %new_select_op
        }
    }

    // select(pred, select(pred, a, b), c) => select(pred, a, c)
    pdl.pattern @RedundantSelectTrue : benefit(0) {
        %i1 = pdl.type : i1
        %type = pdl.type : !transfer.integer<8>

        %pred = pdl.operand : %i1
        %a = pdl.operand : %type
        %b = pdl.operand : %type
        %c = pdl.operand : %type

        %select_op = pdl.operation "arith.select"(%pred, %a, %b : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %select = pdl.result 0 of %select_op

        %select_op2 = pdl.operation "arith.select"(%pred, %select, %c : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %select_op2 {
            %new_select_op = pdl.operation "arith.select"(%pred, %a, %c : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %select_op2 with %new_select_op
        }
    }

    // select(pred, a, select(pred, b, c)) => select(pred, a, c)
    pdl.pattern @RedundantSelectFalse : benefit(0) {
        %i1 = pdl.type : i1
        %type = pdl.type : !transfer.integer<8>

        %pred = pdl.operand : %i1
        %a = pdl.operand : %type
        %b = pdl.operand : %type
        %c = pdl.operand : %type

        %select_op = pdl.operation "arith.select"(%pred, %b, %c : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)
        %select = pdl.result 0 of %select_op

        %select_op2 = pdl.operation "arith.select"(%pred, %a, %select : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %select_op2 {
            %new_select_op = pdl.operation "arith.select"(%pred, %a, %c : !pdl.value, !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %select_op2 with %new_select_op
        }
    }
}

// CHECK-NOT: UNSOUND
