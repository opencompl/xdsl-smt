// RUN: xdsl-smt "%s" --circt -p=pdl-to-smt -t smt | z3 -in | filecheck "%s"

// or(x, y) -> or(y, x)

builtin.module {
  pdl.pattern : benefit(1) {
    %i1 = pdl.type : i1
    %i9 = pdl.type : i9
    %attr0_i64 = pdl.attribute = 0 : i64
    %attr1_i9 = pdl.attribute = -1 : i9

    %arg0 = pdl.operand : %i1
    %arg1 = pdl.operand : %i1
    %arg2 = pdl.operand : %i1
    %arg3 = pdl.operand : %i1
    %arg4 = pdl.operand : %i1
    %arg5 = pdl.operand : %i1
    %arg6 = pdl.operand : %i1
    %arg7 = pdl.operand : %i1
    %arg8 = pdl.operand : %i1

    %concat = pdl.operation "comb.concat" (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 : !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%i9 : !pdl.type)
    %0 = pdl.result 0 of %concat

    %constant = pdl.operation "hw.constant" {"value" = %attr1_i9} -> (%i9 : !pdl.type)
    %m1_i9 = pdl.result 0 of %constant

    %eq = pdl.operation "comb.icmp" (%0, %m1_i9 : !pdl.value, !pdl.value) {"predicate" = %attr0_i64} -> (%i1 : !pdl.type)

    pdl.rewrite %eq {
      %and = pdl.operation "comb.and" (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8 : !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.value) -> (%i1 : !pdl.type)
      pdl.replace %eq with %and
    }
  }
}


// CHECK: unsat
