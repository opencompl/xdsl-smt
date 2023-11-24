// RUN: xdsl-smt "%s" -t mlir -p=pdl-to-smt,canonicalize-smt | filecheck "%s"
// XFAIL: *

builtin.module {
    pdl.pattern @add_constant_fold : benefit(0) {
        %type = pdl.type : i32

        %c0 = pdl.attribute : %type
        %c1 = pdl.attribute : %type

        %constant0 = pdl.operation "arith.constant" {"value" = %c0} -> (%type : !pdl.type)
        %constant1 = pdl.operation "arith.constant" {"value" = %c1} -> (%type : !pdl.type)

        %lhs = pdl.result 0 of %constant0
        %rhs = pdl.result 0 of %constant1

        %add = pdl.operation "arith.add" (%lhs, %rhs : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add {
            %res = pdl.apply_native_rewrite "addi"(%c0, %c1 : !pdl.attribute, !pdl.attribute) : !pdl.attribute
            %res_constant = pdl.operation "arith.constant" {"value" = %res} -> (%type : !pdl.type)
            pdl.replace %add with %res_constant
        }
    }
}
