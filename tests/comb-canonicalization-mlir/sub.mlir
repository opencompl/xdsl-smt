// RUN: verify-pdl "%s" -opt | filecheck "%s"

// extracts only of sub(...) -> sub(extract()...)

// sub(x, cst) -> add(x, -cst)
pdl.pattern @SubCst : benefit(0) {
    %t = pdl.type : !transfer.integer
    %x = pdl.operand : %t

    %cst_attr = pdl.attribute : %t
    %cst_op = pdl.operation "hw.constant" {"value" = %cst_attr} -> (%t : !pdl.type)
    %cst = pdl.result 0 of %cst_op

    %sub_op = pdl.operation "comb.sub"(%x, %cst : !pdl.value, !pdl.value) -> (%t : !pdl.type)

    pdl.rewrite %sub_op {
        %zero_attr = pdl.apply_native_rewrite "get_zero_attr"(%t : !pdl.type) : !pdl.attribute
        %neg_attr = pdl.apply_native_rewrite "subi"(%zero_attr, %cst_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute

        %neg_op = pdl.operation "hw.constant" {"value" = %neg_attr} -> (%t : !pdl.type)
        %neg = pdl.result 0 of %neg_op

        %add_op = pdl.operation "comb.add"(%x, %neg : !pdl.value, !pdl.value) -> (%t : !pdl.type)

        pdl.replace %sub_op with %add_op
    }
}

// CHECK: All patterns are sound